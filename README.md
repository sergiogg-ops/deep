# Visualization
This git contains tools for evaluation and its posterior visualization of Machine Translation (MT) systems. 

## Evaluation
The evaluation of the systems can be done using the `eval.py` script. Each system must be dockerized and prepared to be runned appropiately. It must read the `data/source.sgm` file and write the corresponding translations in the `data/predictions.sgm` file of the docker container. The `eval.py` script will read the predictions, store them in a directory, evaluate each one with the appropriate metrics and clusterize the submissions. 

The clusterization algorithm is based in the significance of the difference in the metrics of the predictions. Thus, once the submissions are sorted by BLEU the significance of the differences between consecutive submissions are assesed. If they are not significant enough those submissions will be part of the same cluster.

The results of the evaluation are stored in a `.csv` file with the specified name.

## Visualization
The visualization is performed using a Streamlit web app. This is the command to launch it:

```
$ streamlit run display.py [OPTIONS] display.py <evaluation_file.csv> [ARGS]
```
![alt text](images/screenshot.png)
