"""CLI харнесса. Точка входа: scripts/run_eval.py (тонкий шим) либо `python -m evaluation.cli`.

  run          — прогнать сьют, записать evaluation/runs/<...>/.
  list-suites  — перечислить доступные сьюты (evaluation/suites/*.yaml) с числом кейсов.
  report       — сравнить прогоны (evaluation/reporting/).

Judge (faithfulness/coverage/answer_relevancy) подключается к `run` только явным
--judge-model — без него сьют даёт только бесплатные детерминированные checks. Отдельный
--judge-model, а не модель самого агента: харнесс намеренно не предлагает "same as main
model" шорткот (см. evaluation/metrics/judge.py) — самосудейство при сравнении 4B/30B
не значит ничего.
"""
import argparse
import logging
import sys

from evaluation.agent_factory import build_agent_for_eval, build_llm_provider
from evaluation.checks.base import CheckContext
from evaluation.config import build_app_config
from evaluation.dataset.loader import SUITES_DIR, load_suite_by_name
from evaluation.metrics.cache import JudgeCache
from evaluation.metrics.judge import JudgeConfig, build_judge
from evaluation.metrics_runner import MetricsRunner
from evaluation.runlog.run_writer import RunWriter
from evaluation.runner import ExperimentRunner, summarize_outcomes
from evaluation.tracing.provider_wrapper import RecordingProvider

logger = logging.getLogger(__name__)


def _cmd_list_suites(args: argparse.Namespace) -> int:
    if not SUITES_DIR.exists():
        print(f"Директория сьютов не найдена: {SUITES_DIR}")
        return 1
    for path in sorted(SUITES_DIR.glob("*.yaml")):
        suite = load_suite_by_name(path.stem)
        print(f"{suite.name:24} entry={suite.entry:16} n_cases={len(suite.cases):4}  {suite.description.strip()[:60]}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    suite = load_suite_by_name(args.suite)
    config = build_app_config(llm_backend=args.llm_backend, model=args.model)

    llm = build_llm_provider(config, record_llm_io=args.record_llm_io)
    agent = build_agent_for_eval(config, llm, offline=args.offline, use_rewriter=not args.no_rewriter)

    context = CheckContext(tokenizer=agent.processor.tokenizer, app_config=config, node_gen=config.node_gen)

    label = args.label or config.mlx_model.split("/")[-1] if config.llm_backend == "mlx" else config.openrouter_model
    writer = RunWriter(
        suite=suite.name,
        label=label,
        app_config=config,
        node_gen=config.node_gen,
        save_artifacts=args.save_artifacts,
        cli_argv=sys.argv,
        extra_manifest={"offline": {"mode": args.offline}, "rewriter": {"enabled": not args.no_rewriter}},
    )

    console_handler = logging.FileHandler(writer.run_dir / "console.log", encoding="utf-8")
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(console_handler)

    recording_provider = llm if isinstance(llm, RecordingProvider) else None

    metrics_runner = None
    if args.judge_model:
        judge_config = JudgeConfig(backend=args.judge_backend, model=args.judge_model)
        judge = build_judge(judge_config)
        cache = JudgeCache()
        needs_embed = any(
            "answer_relevancy" in metric_list
            for scopes in suite.metrics.values()
            for metric_list in scopes.values()
        )
        embed_model = None
        if needs_embed:
            from sentence_transformers import SentenceTransformer

            embed_model = SentenceTransformer(judge_config.embed_model_name)
        metrics_runner = MetricsRunner(judge, cache, judge_model_name=args.judge_model, embed_model=embed_model)
        writer.set_suite_info(judge_model=args.judge_model, judge_backend=args.judge_backend)
        logger.info("Judge подключён: backend=%s model=%s", args.judge_backend, args.judge_model)
    elif suite.metrics:
        logger.warning(
            "Сьют %s объявляет метрики (%s), но --judge-model не передан — будут посчитаны только checks",
            suite.name, list(suite.metrics.keys()),
        )

    runner = ExperimentRunner(agent, context, writer, recording_provider=recording_provider, metrics_runner=metrics_runner)

    status = "completed"
    try:
        outcomes = runner.run_suite(suite, limit=args.limit)
        summary = summarize_outcomes(outcomes)
    except Exception:
        logger.exception("Прогон сьюта %s упал", suite.name)
        status, summary = "failed", {}
    finally:
        writer.finalize(status=status, summary=summary)
        logging.getLogger().removeHandler(console_handler)

    print(f"\nЗавершено: {writer.run_dir}")
    if summary:
        print(
            f"Кейсов: {summary['n_cases']} (ok={summary['n_ok']}, error={summary['n_errored']}); "
            f"проверок: passed={summary['total_checks_passed']} "
            f"failed={summary['total_checks_failed']} warned={summary['total_checks_warned']}"
        )
    return 0 if status == "completed" else 1


def _cmd_report(args: argparse.Namespace) -> int:
    from evaluation.reporting import assert_comparable, compare_runs, list_runs, load_run, render_markdown
    from evaluation.reporting.compare import NotComparableError

    if args.list:
        for run_id in list_runs(suite=args.suite):
            print(run_id)
        return 0

    if not args.baseline or not args.candidate:
        print("Нужны --baseline и --candidate (или --list для перечисления прогонов)")
        return 1

    baseline = load_run(args.baseline)
    candidate = load_run(args.candidate)
    try:
        assert_comparable([baseline, candidate])
    except NotComparableError as exc:
        print(f"Прогоны несравнимы: {exc}")
        return 1

    rows = compare_runs(baseline, candidate)
    print(render_markdown(rows, label_a=args.baseline_label or baseline.run_id, label_b=args.candidate_label or candidate.run_id))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run_eval", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Прогнать сьют")
    p_run.add_argument("--suite", required=True, help="Имя сьюта (evaluation/suites/<name>.yaml)")
    p_run.add_argument("--offline", action="store_true", help="InMemoryArticleStore/FrozenSearchClient вместо сети")
    p_run.add_argument(
        "--no-rewriter", action="store_true",
        help="Отключить QueryRewriter (эвристика ti:/all: вместо LLM-плана) — для сравнения recall@5 до/после (E2)",
    )
    p_run.add_argument("--limit", type=int, default=None, help="Ограничить число кейсов (для быстрой проверки)")
    p_run.add_argument("--label", default=None, help="Метка прогона (по умолчанию — имя модели)")
    p_run.add_argument("--llm-backend", choices=["mlx", "openrouter"], default=None)
    p_run.add_argument("--model", default=None, help="Переопределить модель (mlx_model/openrouter_model)")
    p_run.add_argument("--record-llm-io", action="store_true", help="Писать сами prompt/response в artifacts/")
    p_run.add_argument("--save-artifacts", action="store_true", help="Полные payload'ы узлов в artifacts/<case_id>/")
    p_run.add_argument(
        "--judge-model", default=None,
        help="Модель судьи (faithfulness/coverage/answer_relevancy). Без неё считаются только checks.",
    )
    p_run.add_argument("--judge-backend", choices=["mlx", "openrouter"], default="mlx")
    p_run.set_defaults(func=_cmd_run)

    p_list = sub.add_parser("list-suites", help="Перечислить доступные сьюты")
    p_list.set_defaults(func=_cmd_list_suites)

    p_report = sub.add_parser("report", help="Сравнить два прогона (evaluation/runs/<...>/)")
    p_report.add_argument("--baseline", default=None, help="run_id (имя директории в evaluation/runs/)")
    p_report.add_argument("--candidate", default=None, help="run_id для сравнения с baseline")
    p_report.add_argument("--baseline-label", default=None)
    p_report.add_argument("--candidate-label", default=None)
    p_report.add_argument("--list", action="store_true", help="Перечислить доступные прогоны вместо сравнения")
    p_report.add_argument("--suite", default=None, help="При --list — фильтр по имени сьюта")
    p_report.set_defaults(func=_cmd_report)

    return parser


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
