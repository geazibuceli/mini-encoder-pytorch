# MiniEncoder PyTorch

Classificador educacional de sentimentos para IMDB, com Transformer e baseline de media de embeddings.

## Instalacao

Execute nesta pasta, com Python 3.10 ou superior:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev,data,visualization]"
```

## Uso

```powershell
.venv/Scripts/python.exe -m miniencoder.doctor
.venv/Scripts/python.exe -m miniencoder.prepare_data
.venv/Scripts/python.exe -m miniencoder.train --config configs/quick_test.yaml --checkpoint results/model.pt
.venv/Scripts/python.exe -m miniencoder.evaluate --checkpoint results/model.pt
.venv/Scripts/python.exe -m miniencoder.predict "A wonderful movie" --checkpoint results/model.pt
.venv/Scripts/python.exe -m pytest -q
```

Os dados preparados existentes podem ser usados sem executar `prepare_data` (que baixa IMDB).
O comprimento do modelo e inferido dos metadados dos dados quando `model.max_length` e omitido.
Uma capacidade explicitamente menor e rejeitada antes do treinamento.
O checkpoint salva a epoca de menor perda de validacao, com configuracao do modelo e estados do otimizador e scheduler.
O historico dentro dele termina na epoca selecionada; a CLI imprime os resultados da ultima epoca executada.

Para treinar via API, chame `set_seed(seed)` antes de construir o modelo e passe a mesma seed a `train_model`.
`train_model` nao reinicializa os pesos de modelos existentes. Seu modelo em memoria permanece na ultima epoca;
carregue o checkpoint salvo para usar a melhor epoca. A reproducibilidade entre dispositivos e versoes distintas nao e garantida.

## Estrutura recuperada

O codigo mantido fica em `src/miniencoder`. Ele foi recuperado da copia que existia em `build/lib`.
Os testes em `tests` sao novos testes de regressao: os fontes dos testes originais nao estavam presentes.
`build/lib` e os caches antigos foram preservados e nao devem ser usados como fonte de desenvolvimento.
Os checkpoints e relatorios antigos foram preservados; as correcoes nao retreinam esses modelos automaticamente.
