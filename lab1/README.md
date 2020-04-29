# Лабораторная работа 1

## Данные
Картинки и отчет о работе лежат на 
[google drive](https://drive.google.com/drive/folders/1cyS5W3SEnRyq2WPHrVAD0IrCjFeYITTA?usp=sharing) 
с суффиксом 1.

## How to run
Для запуска программы необходим python 3.7.0. Необходимо установить
пакеты из requirements.txt:
```bash
pip install -r requirements.txt
```

Посмотреть налядно работу функций можно в exploration.ipynb.

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
* --input-file - путь к изображению
* --segmentation - `motley` или `monochrome` в зависимости от входной картинки
* --corner-candidates - число потенциальных углов для приближающего четырехугольника. Рекомендуется 33.
Полученный результат сохраняется в output.png.
