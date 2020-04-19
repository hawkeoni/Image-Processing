# Обработка и распознавание изображений 2020
Для запуска программы необходим python 3.7.0. Необходимо установить
пакеты из requirements.txt:
```bash
pip install -r requirements.txt
```
Запуск программы осуществляется командой:
```
usage: main.py [-h] --input-file INPUT_FILE --segmentation SEGMENTATION
               [--corner-candidates CORNER_CANDIDATES]

optional arguments:
  -h, --help            show this help message and exit
  --input-file INPUT_FILE
                        Input image.
  --segmentation SEGMENTATION
                        Segmentation type.
  --corner-candidates CORNER_CANDIDATES
                        Number of candidate corners for best rectangle
                        approximation.
```
где:
* --input-file - путь к изображениею
* --segmentation - `motley` или `monochrome` в зависимости от входной картинки
* --corner-candidates - число потенциальных углов для приближающего четырехугольника. Рекомендуется 33.
