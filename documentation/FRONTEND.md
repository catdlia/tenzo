# 🛰 Модуль: Frontend & Model Exporters

Цей модуль відповідає за імпорт навчених моделей (PyTorch, SafeTensors, GGUF, AutoGPTQ, AutoAWQ, EXL2, ONNX) та їх перетворення у внутрішнє представлення Tenzo MLIR і бінарні ваги.

## 📂 Структура файлів
*   **`export_bitnet.py`**: Експортер BitNet 1.58b з Hugging Face у `model.mlir` та упаковані бінарні ваги `weights.bin`.
*   **`fx_to_mlir.py`**: PyTorch FX Graph capture для автоматичної генерації MLIR графа `tenzo.ternary_pack` та `tenzo.packed_attention`.
*   **`export_gguf.py` / `export_llama.py`**: Експорт моделей GGUF (`Q4_0`, `Q8_0`) та LLaMA.
*   **`export_gptq.py` / `export_awq.py` / `export_exl2.py`**: Експортери для 4-бітних та змінних бітрейтів (AutoGPTQ, AutoAWQ, ExLlamaV2).
*   **`qat.py`**: Quantization-Aware Training (QAT) утиліти та фейкова квантизація.
*   **`onnx_to_mlir.py`**: Класичний ONNX-конвертер.
*   **`weight_utils.py`**: Утиліти для роботи з бінарними даними та пакуванням тензорів.

## 🛠 Ключові функції

### `onnx_to_mlir(onnx_path, output_path)`
**Принцип роботи:**
1.  **Topological Sort**: Модель ONNX розглядається як граф. Функція проходить по вузлах у порядку їх виконання.
2.  **Pattern Matching (Fusion)**: Під час обходу функція заглядає на один крок вперед. Якщо вона бачить `Add` за яким йде `Relu`, вона генерує одну операцію `tenzo.fused_add_relu`.
3.  **Shape Inference**: Використовує бібліотеку ONNX для обчислення розмірів тензорів, що дозволяє створювати точні MLIR типи (наприклад, `tensor<1x10xf32>`).

### `resolve_inputs(...)`
**Функція:** Визначає, чи є вхідний тензор вагою (initializer) чи результатом попередньої операції.
*   Якщо це вага: генерує код для завантаження з глобальної пам'яті (`memref.get_global`).
*   Якщо це результат: використовує SSA-ідентифікатор (наприклад, `%v5`).

## 🔄 Пайплайн конвертації
1.  **Завантаження**: `onnx.load()`.
2.  **Аналіз**: `shape_inference.infer_shapes()`.
3.  **Збереження ваг**: Кожен `initializer` зберігається у `weights/name.bin`.
4.  **Емісія MLIR**: Запис текстового представлення діалекту Tenzo.

## 💡 Використання
```bash
make onnx-convert MODEL=my_model.onnx
```
Згенерований файл `.mlir` можна відкрити текстовим редактором для перевірки структури мережі.
