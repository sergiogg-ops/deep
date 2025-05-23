import argparse
import os
import sacrebleu
import fastwer
import pandas as pd
import lxml.etree as ET
from tqdm import tqdm
from time import time
from art import aggregators, scores, significance_tests

def read_parameters():
    parser = argparse.ArgumentParser(description='Evaluates the participant dockerized models. They need to output the translations of the source file in SGM format.')
    parser.add_argument('source', type=str, help='Path to the sources file')
    parser.add_argument('reference', type=str, nargs='+', help='Path to the references file(s)')
    parser.add_argument('--systems', type=str, help='Path to the directory that contains all the dockerized systems. If provided, the systems will be run and the translations will be evaluated. If not provided, the translations will be read from the dir_preds directory.')
    parser.add_argument('--dir_preds', type=str, default='translations', help='Name of directory with the translation files')
    parser.add_argument('--baselines', type=str, nargs='+', help='List of baseline systems to be evaluated. Must be included among the rest of the systems')
    parser.add_argument('--output', type=str, default='results.csv', help='Path to the file that will store the leaderboard')
    parser.add_argument('-a','--append', action='store_true', help='Append the results to the output file')
    parser.add_argument('--metrics', type=str, nargs='+', default=['bleu', 'ter'], choices=['bleu', 'ter'], help='List of metrics to be used (default: BLEU and TER)')
    parser.add_argument('--trials', type=int, default=10000, help='Number of trials for the ART (default: 10000)')
    parser.add_argument('--p_value', type=float, default=0.05, help='P-value for the ART (default: 0.05)')
    args = parser.parse_args()
    return args

def get_bleu(x, y):
    '''
    Compute the BLEU score between two lists of segments.
    Args:
        x: list of translated segments
        y: list of reference segments
    Returns:
        bleu: BLEU score
    '''
    bleu = sacrebleu.corpus_bleu(x, y).score
    return bleu
def get_ter(x, y):
    '''
    Compute the TER score between two lists of segments.
    Args:
        x: list of translated segments
        y: list of reference segments
    Returns:
        ter: TER score
    '''
    ter = sacrebleu.corpus_ter(x, y).score
    return ter

def check_paramaters(args):
    # Check if the systems directory exists and get the list of models
    if args.systems:
        try:
            models = os.listdir(args.systems)
        except FileNotFoundError:
            raise FileNotFoundError(f"Directory {args.systems} not found.")
    else: 
        models = []
    # Check if the source file exists
    try:
        os.path.exists(args.source)
    except FileNotFoundError:
        raise FileNotFoundError(f"Source file {args.source} not found.")
    # Check if the reference file exists
    try:
        refs = [parse_xml(file) for file in args.reference]
    except FileNotFoundError:
        raise FileNotFoundError(f"Reference file {args.reference} not found.")
    if not os.path.exists(args.dir_preds):
        os.makedirs(args.dir_preds)
    # Obtain the functions for the metrics
    metrics = {}
    for m in args.metrics:
        if m == 'bleu':
            metrics[m] = get_bleu
        elif m == 'ter':
            metrics[m] = get_ter
    return models, refs, metrics

def parse_xml(file):
    '''
    Parse the XML file and return the segments.
    Args:
        file: path to the XML file
    Returns:
        segments: list of segments
    '''
    try:
        tree = ET.parse(file,  ET.XMLParser(recover=True))
    except IOError:
        raise IOError(f"File {file} not found.")
    except SyntaxError:
        raise SyntaxError(f"File {file} is not well-formed.")
    
    root = tree.getroot()
    segments = {}
    for doc in root: #tree.getiterator(tag='doc'): 
        segs = []
        for tag in doc:
            if tag.tag == 'hl' or tag.tag == 'p':
                for seg in tag:
                    if seg.text is None:
                        segs.append('')
                    else:
                        segs.append(seg.text.strip())
            elif tag.tag == 'seg':
                if tag.text is None:
                    segs.append('')
                else:
                    segs.append(tag.text.strip())
        if len(segs) > 0:
            segments[doc.attrib['docid']] = segs
    order = sorted(segments.keys())
    segments = [segments[docid] for docid in order]
    return segments[0] if len(segments) == 1 else segments

def parse_moses(file):
    with open(file,'r') as f:
        segments = f.readlines()
    return segments

def evaluate(preds, refs, metrics):
    '''
    Evaluate the translations using BLEU and TER.
    Args:
        preds: list of translated segments
        refs: list of reference segments
        metrics: list of metrics (functions) to be used (default: BLEU and TER)
    Returns:
        global_bleu: BLEU score
        global_ter: TER score
        metrics: list of tuples with BLEU and TER scores for each segment
    '''
    scores = []
    n_refs = len(refs)
    keys = list(metrics.keys())
    keys.sort()
    for i in range(len(preds)):
        score = []
        for k in keys:
            func = metrics[k]
            score.append(func([preds[i]], [refs[j][i] for j in range(n_refs)]))
        scores.append(score)
    global_metrics = {k:metrics[k](preds, refs) for k in keys}
    return global_metrics, scores

def assess_differences(a_metrics, b_metrics, trials, p_value):
    '''
    Assess the differences between two sets of metrics using Approximate Randomization Test.
    Args:
        a_metrics: list of metrics for system A (can be a list of lists)
        b_metrics: list of metrics for system B (can be a list of lists)
        trials: number of trials for the test
        p_value: p-value for the test 
    Returns:
        True if the difference is significant, False otherwise
    '''
    test = significance_tests.ApproximateRandomizationTest(
        scores.Scores([scores.Score(m) for m in a_metrics]),
        scores.Scores([scores.Score(m) for m in b_metrics]),
        aggregators.average,
        trials=trials)

    return test.run() < p_value

def run_tests(models, models_dir, source, dir_preds):
    '''
    Translate the sources with all the systems.

    Args:
        models: list of models to be evaluated
        models_dir: path to the directory that contains all the dockerized systems
        source: path to the sources file
        dir_preds: path to the directory that will store the translation files
    Returns:
        run_times: list of times taken to run each model
    '''
    run_times = []
    for model in tqdm(models, desc="Running participants"):
        output_name = os.path.join(dir_preds, model + '.sgm')
        print(f"Building {model}")
        os.system(f"docker build -t {model} {models_dir}/{model}")
        os.system(f"touch {output_name}")
        print(f"Running {model}")
        ini = time()
        os.system(f"docker run --rm -v {os.path.abspath(source)}:/data/source.sgm -v {os.path.abspath(output_name)}:/data/predictions.sgm --gpus all {model} ")
        fin = time()
        run_times.append(fin - ini)
        os.system(f"docker rmi {model}")
    print("Cleaning up...")
    os.system("docker system prune -f")
    os.system("docker volume prune -f")
    return run_times

def main():
    args = read_parameters()
    models, refs, metrics = check_paramaters(args)
    
    # Translate the sources
    if models:
        models.sort()
        run_times = run_tests(models, args.systems, args.source, args.dir_preds)
    else:
        run_times = [0] * len(os.listdir(args.dir_preds))
    
    fields = ['name'] + list(metrics.keys()) + ['cluster', 'time', 'datetime', 'bench', 'metrics','comment']
    register = pd.DataFrame(columns=fields)
    # Evaluate the translations
    predictions = os.listdir(args.dir_preds)
    predictions.sort()
    for filename in tqdm(predictions, desc="Evaluating"):
        try:
            preds = parse_xml(os.path.join(args.dir_preds, filename))
            global_scores, scores = evaluate(preds, refs , metrics)
            for k in global_scores.keys():
                global_scores[k] = [global_scores[k]]
            global_scores['metrics'] = [scores] # sentence scores to asses the significance of the differences
        except Exception as e:
            global_scores = {k: [None] for k in metrics.keys()}
            global_scores['metrics'] = [None]
            print(f"Error evaluating {filename}: {e}")
        global_scores['name'] = [os.path.basename(filename).replace('.sgm','')]
        global_scores['datetime'] = [pd.Timestamp.now()]
        global_scores['bench'] = [os.path.basename(args.source) + '-' + os.path.basename(args.reference)]
        global_scores['comment'] = ['Baseline' if global_scores['name'][0] in args.baselines else '']
        register = pd.concat([register, pd.DataFrame(global_scores)],ignore_index=True)
    
    # Sort the participants by BLEU score
    register['time'] = run_times
    register = register.sort_values(by=['bleu'], ascending=False)
    register['position'] = [i+1 for i in range(len(register))]

    # Check the significance between the systems
    cluster_id = int(1)
    clusters = []
    for i in tqdm(range(len(register)-1),desc="Clustering"):
        this_row = register.iloc[i]
        next_row = register.iloc[i+1]
        if this_row['bleu'] is None or next_row['bleu'] is None:
            break
        diff = assess_differences(this_row['metrics'], next_row['metrics'],trials=args.trials, p_value=args.p_value)
        clusters.append(cluster_id)
        if diff:
            cluster_id += 1
    clusters.append(cluster_id)
    if len(clusters) < len(register):
        clusters += [cluster_id + 1] * (len(register) - len(clusters))
    register['cluster'] = clusters

    # Save the results
    register = register.drop(columns=['metrics'])
    if args.append:
        register.to_csv(args.output, mode='a', header=False, index=False)
    else:
        register.to_csv(args.output, mode='w', header=True, index=False)
    #register.to_csv(args.output, index=False)
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()