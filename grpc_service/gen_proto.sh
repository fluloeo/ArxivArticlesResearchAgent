#!/usr/bin/env bash
# Регенерирует protobuf/gRPC стабы из arxiv_agent.proto.
# Запускать из корня репозитория: bash grpc_service/gen_proto.sh
set -euo pipefail

cd "$(dirname "$0")/.."

python3 -m grpc_tools.protoc \
    -I grpc_service/proto \
    --python_out=grpc_service/generated \
    --grpc_python_out=grpc_service/generated \
    grpc_service/proto/arxiv_agent.proto

# grpc_tools.protoc генерирует абсолютный `import arxiv_agent_pb2 as ...` в _grpc.py,
# что не резолвится как относительный импорт внутри пакета grpc_service.generated —
# переписываем на relative import.
sed -i.bak 's/^import arxiv_agent_pb2 as/from . import arxiv_agent_pb2 as/' grpc_service/generated/arxiv_agent_pb2_grpc.py
rm -f grpc_service/generated/arxiv_agent_pb2_grpc.py.bak

echo "Стабы сгенерированы в grpc_service/generated/"
