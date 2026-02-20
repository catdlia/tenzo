
---

### ⚡ Як запускати швидше (Правильний Workflow)

Ти щоразу запускаєш `docker run`, що створює новий контейнер. Це додає 1-2 секунди затримки, але не 10 хвилин.
Щоб працювати швидко (як на локалці), використовуй **Інтерактивний Режим**.

Ось оновлений файл `CLI_GUIDE.md`, який пропонує **правильний workflow** для розробки.

```markdown
# 🛠️ Tenzo CLI Guide: Інструкція розробника

Цей документ описує найшвидший спосіб розробки та тестування Tenzo.

## ⚡ 1. Режим Розробника (Найшвидший)

Замість того, щоб запускати `docker run` для кожної команди, зайдіть у контейнер один раз і працюйте всередині. Це економить час на старт контейнера і зберігає історію команд.

**Крок 1: Зайти в контейнер**
```bash
docker run --rm -it -v $(pwd):/app -w /app tenzo-dev:latest bash

```

**Крок 2: Всередині контейнера (збираємо і запускаємо)**
Тепер ви в Linux-терміналі. Всі команди виконуються миттєво.

* **Збірка (один раз налаштувати, потім тільки ninja):**
```bash
mkdir -p build_e2e && cd build_e2e
cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release

```


* **Перезбірка + Запуск (ваша основна команда):**
```bash
ninja tenzo-cli && ./tenzo-cli gemm-e2e

```



---

## 🎮 2. Основні Команди (Running Tests)

Запускати **всередині** папки `build_e2e/`.

### 🏆 `gemm-e2e` (Головний Тест)

Повний пайплайн: Packing + Loops + Micro-kernel.

```bash
./tenzo-cli gemm-e2e

АБО
docker run --rm -v $(pwd):/app -w /app tenzo-dev:latest ./build_e2e/tenzo-cli gemm-e2e

```

*Якщо зависає:* Використовуйте `timeout 10s ./tenzo-cli gemm-e2e` щоб не вішати термінал.

### 🔬 `micro_bench` (Чисте Ядро)

Перевірка пікової швидкості AVX2 (без пакування).
**Треба збирати окремо:** `ninja micro_bench`

```bash
./micro_bench

АБО
docker run --rm -v $(pwd):/app -w /app tenzo-dev:latest bash -c "
    cd build_e2e && 
    cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release -DUSE_MLIR_KERNEL=ON && 
    ninja micro_bench && 
    ./micro_bench"
    
```

### 📦 `packing` (Тест Пам'яті)

Тестує лише швидкість перевпорядкування даних.

```bash
./tenzo-cli packing

АБО
docker run --rm -v $(pwd):/app -w /app tenzo-dev:latest ./build_e2e/tenzo-cli packing

```

**⚠️ Увага:** Цей тест має проходити за < 1 секунду. Якщо він триває довше — у вас нескінченний цикл у `PackingPass.cpp`.

---

## 🎛️ 3. Як перевірити, чому зависло?

Якщо команда "думає" вічність, зробіть так (всередині контейнера):

1. Встановіть налагоджувач:
```bash
apt-get update && apt-get install -y gdb

```


2. Запустіть під налагоджувачем:
```bash
gdb --args ./tenzo-cli packing

```


3. У gdb напишіть `run`. Коли зависне, натисніть `Ctrl+C`, а потім напишіть `bt` (backtrace). Це покаже рядок коду, де зациклилась програма.

