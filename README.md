# DEEP: Docker-based Evaluation and Execution Platform
🚨⚠️🚧 **Experimental branch** 🚧⚠️🚨
![Screenshot of the visual interface](images/screenshot.png)
This git contains a pipeline for automated execution of systems and evaluation of Machine Translation (MT) and Optical Character Recognition (OCR) systems. After the evaluation we also provide a visualization web-app to analyse the results.

Before using the software it is important to create a conda environment and run the `setup.sh` script:
```bash
conda env create -f environment.yaml
sh setup.sh
```
Otherwise some functionalities won't work. Furthermore, to use the metric BEER the system need to have [java](https://www.java.com/en/download/manual.jsp) installed.

## TODOs
- [x] Image generation evaluation.
  - [x] SSIM
  - [x] FID
- [ ] Image generation execution.

## Evaluation
The evaluation of the systems can be done using the `eval.py` script: 
```
usage: python eval.py [-h] [--source SOURCE [SOURCE ...]] [--systems SYSTEMS]
               [--dir_preds DIR_PREDS] [--baselines BASELINES [BASELINES ...]]
               [--output OUTPUT] [-a]
               --metrics {bleu,ter,chrf,wer,bwer,beer,fid,ssim,iou} [{bleu,ter,chrf,wer,bwer,beer,fid,ssim,iou} ...]
               [--main_metric MAIN_METRIC] [--ascending] [--trials TRIALS]
               [--p_value P_VALUE] --task {mt,ocr,img,t_det} --subtask SUBTASK
               reference [reference ...]

Evaluates the participant dockerized models. They need to produce the hypotheses of the
source file in the same format as the reference(s) file(s).

positional arguments:
  reference             Path to the references file(s)

options:
  -h, --help            show this help message and exit
  --source SOURCE [SOURCE ...]
                        Path to the sources files
  --systems SYSTEMS     Path to the directory that contains all the dockerized systems. If
                        provided, the systems will be run and the translations will be
                        evaluated. If not provided, the translations will be read from the
                        dir_preds directory.
  --dir_preds DIR_PREDS
                        Name of directory with the translation files
  --baselines BASELINES [BASELINES ...]
                        List of baseline systems to be evaluated. Must be included among
                        the rest of the systems
  --output OUTPUT       Path to the file that will store the leaderboard
  -a, --append          Append the results to the output file
  --metrics {bleu,ter,chrf,wer,bwer,beer,fid,ssim,iou} [{bleu,ter,chrf,wer,bwer,beer,fid,ssim,iou} ...]
                        List of metrics to be used (default: BLEU and TER)
  --main_metric MAIN_METRIC
                        Main metric to sort the leaderboard (default: bleu for MT and bwer
                        for DR)
  --ascending           Sort the leaderboard in ascending order (default: descending)
  --trials TRIALS       Number of trials for the ART (default: 10000)
  --p_value P_VALUE     P-value for the ART (default: 0.05)
  --task {mt,ocr,img,t_det}
                        Task to be evaluated: mt (machine translation), ocr (optical
                        character recognition), img (image generation), t_det (text
                        detection)
  --subtask SUBTASK     Subtask to be evaluated
```
Each system must be dockerized and prepared to be runned appropiately. It must read the `data/source.sgm` file and write the corresponding translations in the `data/predictions.sgm` file of the docker container. The `eval.py` script will read the predictions, store them in a directory, evaluate each one with the specified metrics and clusterize the submissions. 

The clusterization algorithm is based in the significance of the difference in the metrics of the predictions. Thus, once the submissions are sorted by BLEU the significance of the differences between consecutive submissions are assesed. If they are not significant enough those submissions will be part of the same cluster.

The results of the evaluation are stored in a `.csv` file with the specified name.

## Visualization
The visualization is performed using a Streamlit web app. This is the command to launch it:
```
usage: streamlit run display.py [-h] filename

Display evaluation results

positional arguments:
  filename    Path to the CSV file containing evaluation results

options:
  -h, --help  show this help message and exit
```
## Demo
A demo/tutorial for the machine translation (MT) task is available in the `demo` folder. To launch the execution+evaluation proccess, place yourself in the cloned repository directory and run the command:
```bash
python eval.py demo/test_set/newstestB2020-ende-ref.de.sgm --source demo/test_set/newstestB2020-ende-src.en.sgm --systems demo/proposals/ --dir_preds demo/hypotheses/ --output demo/results.csv --metrics bleu ter chrf --task mt --subtask demo
```
We have made available also the hypotheses that the proposals should generate en case that the user wants to avoid running the execution. They can execute just the evaluation by running the same command without the `--systems demo/proposals/` part. Even if their intention is just to try the visualization, the `results.csv` file is also available. To execute the visualization tool the user just need to run the following command:
```bash
streamlit run display.py demo/results.csv
```
Copy the correspoding url to your favorite browser and navigate to it.

## Git guide
- `metrics.py`: contains the functions needed for scoring the hypotheses and runing automated randomized tests to check the statistical significance.
- `eval.py`: script that automates the execution and evaluation of NLP systems.
- `display.py`: script that contains the web-app to analyse the results of the evaluation.
- `setup.sh`: script that installs all the dependencies to execute the rest of functions of the application.
- `demo.csv`: file that contains the results of an example evaluation.
- `requirements.txt`: contains the python dependencies to execute the different parts of the pipeline.
- `demo`: directory with the resources for the demo/tutorial. It contains the dockerized MT systems in the `demo/proposals` directory, the hypotheses they produce in the `demo/hypothesis` directory and the results of their evaluation in the `demo/results.csv` file. The test dataset is stored in the `demo/test_set` directory.