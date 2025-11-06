#!/bin/bash

python eval.py prueba_ocr/refs/AIDWLU.xml --dir_preds prueba_ocr/AIDWLU/ --output prueba_ocr/AIDWLU.csv --metrics wer bwer --task dr --subtask dr
python eval.py prueba_ocr/refs/AIGDTW.xml --dir_preds prueba_ocr/AIGDTW/ --output prueba_ocr/AIGDTW.csv --metrics wer bwer --task dr --subtask dr
python eval.py prueba_ocr/refs/AJBJWO.xml --dir_preds prueba_ocr/AJBJWO/ --output prueba_ocr/AJBJWO.csv --metrics wer bwer --task dr --subtask dr
python eval.py prueba_ocr/refs/ARBXCK.xml --dir_preds prueba_ocr/ARBXCK/ --output prueba_ocr/ARBXCK.csv --metrics wer bwer --task dr --subtask dr