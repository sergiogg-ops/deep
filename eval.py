import argparse
import os
import sacrebleu
import fastwer
import pandas as pd
import lxml.etree as ET
import subprocess
import sys
from tqdm import tqdm
from time import time
from re import sub, findall
from unicodedata import normalize
from art import aggregators, scores, significance_tests

def read_parameters():
    parser = argparse.ArgumentParser(description='Evaluates the participant dockerized models. They need to output the translations of the source file in SGM format.')
    parser.add_argument('reference', type=str, nargs='+', help='Path to the references file(s)')
    parser.add_argument('--source', type=str, nargs='+', help='Path to the sources files')
    parser.add_argument('--systems', type=str, help='Path to the directory that contains all the dockerized systems. If provided, the systems will be run and the translations will be evaluated. If not provided, the translations will be read from the dir_preds directory.')
    parser.add_argument('--dir_preds', type=str, default='translations', help='Name of directory with the translation files')
    parser.add_argument('--baselines', type=str, nargs='+', default=[], help='List of baseline systems to be evaluated. Must be included among the rest of the systems')
    parser.add_argument('--output', type=str, default='results.csv', help='Path to the file that will store the leaderboard')
    parser.add_argument('-a','--append', action='store_true', help='Append the results to the output file')
    parser.add_argument('--metrics', type=str, nargs='+', default=['bleu', 'ter'], choices=['bleu', 'ter','chrf','beer','wer','bwer'], help='List of metrics to be used (default: BLEU and TER)')
    parser.add_argument('--trials', type=int, default=10000, help='Number of trials for the ART (default: 10000)')
    parser.add_argument('--p_value', type=float, default=0.05, help='P-value for the ART (default: 0.05)')
    parser.add_argument('--task', type=str, required=True, choices=['mt','dr'], help='Task to be evaluated: mt (machine translation) or dr (document recognition)')
    parser.add_argument('--subtask', type=str, required=True, help='Subtask to be evaluated')
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

def get_chrf(x, y):
    '''
    Compute the CHR-F score between two lists of segments.
    Args:
        x: list of translated segments
        y: list of reference segments
    Returns:
        chrf: CHR-F score
    '''
    chrf = sacrebleu.corpus_chrf(x, y).score
    return chrf

def get_wer(x, y):
    '''
    Compute the WER score between two lists of segments.
    Args:
        x: list of translated segments
        y: list of lists of reference segments
    Returns:
        wer: WER score
    '''
    if isinstance(y[0], list):
            y = y[0]
    if type(x) != type(y):
        raise TypeError("x and y must be of the same type (list or str)")
    edits, lengths = 0 ,0
    for sent_x, sent_y in zip(x, y):
        sent_x, sent_y = wer_norm(sent_x), wer_norm(sent_y)
        e, l = fastwer.compute(sent_x, sent_y, char_level=False)
        edits += e
        lengths += l
    return 100 * edits / lengths if lengths > 0 else 0

def wer_norm(x):
    x = normalize('NFC', x)
    x = sub(r'\s+',' ',x)
    return x

def get_bwer(x, y):
    '''
    Compute the BWER score between two lists of segments.
    Args:
        x: list of translated segments
        y: list of reference segments
    Returns:
        bwer: BWER score
    '''
    if isinstance(y[0], list):
            y = y[0]
    if type(x) != type(y):
        raise TypeError("x and y must be of the same type (list or str)")
    global_scr = 0
    glob_ref_wl = 0
    for sent_x, sent_y in zip(x, y):
        sent_x, sent_y = wer_norm(sent_x), wer_norm(sent_y)
        scr = fastwer.bagOfWords(sent_x, sent_y, char_level=False)[0]
        dfa = abs(len(sent_x.split()) - len(sent_y.split()))
        global_scr += (scr - dfa) // 2 + dfa
        glob_ref_wl += len(sent_x.split())
    return 100 * global_scr / glob_ref_wl

def get_beer(x, y):
    '''
    Compute the BEER score between two lists of segments.
    Args:
        x: list of translated segments
        y: list of reference segments
    Returns:
        beer: BEER score
    '''
    dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    hyps_file = save_to_file(x)
    refs_files = [save_to_file(ref) for ref in y]

    # Compute BEER
    try:
        process = subprocess.Popen((dir + '/beer_2.0/beer -s ' + hyps_file
                                    + ' -r ' + ':'.join(refs_files)
                                    + ' --printSentScores').split(),
                                   stdout=subprocess.PIPE)
        beer, _ = process.communicate()
    except FileNotFoundError:
        sys.stderr.write('Error: Beer requirement has not been satisfied.\n')
        sys.exit(-1)

    # Delete aux files
    process = subprocess.Popen(('rm ' + hyps_file + ' '
                                + ' '.join(refs_files)).split(), stdout=subprocess.PIPE)
    beer = beer.decode('utf-8')
    beer = [float(s) for s in findall(r"score is ([0-9.]+)", beer)]
    return  sum(beer) / len(beer), beer

def save_to_file(sentences):
    file = '/tmp/' + str(time()) + '_archer'
    with open(file, 'w') as f:
        f.write('\n'.join(sentences))
    return file

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
    if args.source:
        for src in args.source:
            if not os.path.exists(src):
                raise FileNotFoundError(f"Source file {src} not found.")
    # Check if the reference file exists
    try:
        func = parse_xml if args.task == 'mt' else read_dir
        refs = []
        for dir in args.reference:
            refs.append([])
            docs = os.listdir(dir) 
            docs.sort()
            for doc in docs:
                refs[-1].extend(func(os.path.join(dir, doc)))
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
        elif m == 'chrf':
            metrics[m] = get_chrf
        elif m == 'wer':
            metrics[m] = get_wer
        elif m == 'bwer':
            metrics[m] = get_bwer
    return models, refs, metrics

def parse_xml(file):
    try:
        tree = ET.parse(file,  ET.XMLParser(recover=True))
    except IOError:
        raise IOError(f"File {file} not found.")
    except SyntaxError:
        raise SyntaxError(f"File {file} is not well-formed.")
    
    root = tree.getroot()
    segments = []
    for seg in root: #tree.getiterator(tag='doc'): 
        if seg.tag == 'SEG':
            if seg.text is None:
                segments.append('')
            else:
                segments.append(seg.text.strip())
    return segments

def parse_moses(file):
    with open(file,'r') as f:
        segments = f.readlines()
    return segments

def read_dir(dirname):
    '''
    Read the directory and return the text.
    Args:
        dirname: path to the directory
    Returns:
        text: list of segments
    '''
    texts = []
    list_dir = os.listdir(dirname)
    list_dir.sort()
    for filename in list_dir:
        with open(os.path.join(dirname, filename), 'r') as f:
            texts.append(f.readlines()[0])
    return texts

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
    scores = {}
    n_refs = len(refs)
    keys = list(metrics.keys())
    keys.sort()
    for k in keys:
        func = metrics[k]
        scores[k] = []
        for i in range(len(preds)):
            # print('-----------------------------------')
            # print(preds[i])
            # print([type(refs[j][i]) for j in range(n_refs)])
            scores[k].append(func([preds[i]], [refs[j][i] for j in range(n_refs)]))
    global_metrics = {k:metrics[k](preds, refs) for k in keys}
    return global_metrics, scores

def assess_differences(a_scores, b_scores, trials, p_value):
    '''
    Assess the differences between two sets of metrics using Approximate Randomization Test.
    Args:
        a_metrics: list of metrics for system A
        b_metrics: list of metrics for system B
        trials: number of trials for the test
        p_value: p-value for the test 
    Returns:
        True if the difference is significant, False otherwise
    '''
    test = significance_tests.ApproximateRandomizationTest(
        scores.Scores([scores.Score([s]) for s in a_scores]),
        scores.Scores([scores.Score([s]) for s in b_scores]),
        aggregators.average,
        trials=trials)

    run_val = test.run()
    #print(f"Approximate Randomization Test: {run_val} (p-value: {p_value})")
    return run_val < p_value

def run_tests(models, models_dir, sources, dir_preds):
    '''
    Translate the sources with all the systems.

    Args:
        models: list of models to be evaluated
        models_dir: path to the directory that contains all the dockerized systems
        sources: path to the sources file
        dir_preds: path to the directory that will store the translation files
    Returns:
        run_times: list of times taken to run each model
    '''
    run_times = []
    for model in tqdm(models, desc="Running participants"):
        print(f"Building {model}")
        os.system(f"docker build -t {model} {models_dir}/{model}")
        print(f"Running {model}")
        ini = time()
        for src in sources:
            output_name = os.path.join(dir_preds, model + '_' + os.path.basename(src))
            os.system(f"touch {output_name}")
            os.system(f"docker run --rm -v {os.path.abspath(src)}:/data/source.sgm -v {os.path.abspath(output_name)}:/data/predictions.sgm --gpus all {model} ")
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
    # print('############################')
    # print([len(r) for r in refs])
    # print('############################')
    
    # Translate the sources
    if models:
        models.sort()
        run_times = run_tests(models, args.systems, args.source, args.dir_preds)
        run_times = {model: run_times[i] for i, model in enumerate(models)}
    else:
        run_times = {}
    fields = ['name'] + list(metrics.keys()) + ['time', 'datetime', 'metrics', 'comment']
    register = pd.DataFrame(columns=fields)
    # Evaluate the translations
    predictions = os.listdir(args.dir_preds)
    predictions.sort()
    full_preds, models = [], []
    prev_name = ''
    read_func = parse_xml if args.task == 'mt' else read_dir
    for filename in predictions:
        prefix = filename.split('_')[0]
        if prefix != prev_name:
            full_preds.append([])
            models.append(prefix)
        next = read_func(os.path.join(args.dir_preds, filename))
        full_preds[-1].extend(next)
        prev_name = filename.split('_')[0]
    for preds, model in tqdm(zip(full_preds, models), desc="Evaluating",total=len(models)):
        try:
            global_scores, scores = evaluate(preds, refs, metrics)
            for k in global_scores.keys():
                global_scores[k] = [global_scores[k]]
            if 'beer' in args.metrics:
                beer, sent_scr = get_beer(preds, refs)
                global_scores['beer'] = [beer]
                scores['beer'] = sent_scr
            global_scores['metrics'] = [scores] # sentence scores to asses the significance of the differences
        except Exception as e:
            global_scores = {k: [None] for k in metrics.keys()}
            global_scores['metrics'] = [None]
            print(f"Error evaluating {model}: {e}")
        global_scores['name'] = [model]
        global_scores['datetime'] = [pd.Timestamp.now()]
        global_scores['task'] = [args.task]
        global_scores['subtask'] = [args.subtask]
        global_scores['comment'] = ['Baseline' if global_scores['name'][0] in args.baselines else '']
        global_scores['time'] = [run_times.get(global_scores['name'][0], 0)]
        register = pd.concat([register, pd.DataFrame(global_scores)],ignore_index=True)
    
    # Sort the participants by the corresponding score
    main_metric = 'bleu' if args.task == 'mt' else 'bwer'
    register = register.sort_values(by=[main_metric], ascending=False)
    register['position'] = [i+1 for i in range(len(register))]

    # Check the significance between the systems
    for metric in args.metrics:
        cluster_id = int(1)
        clusters = []
        for i in tqdm(range(len(register)-1),desc="Clustering " + metric):
            this_row = register.iloc[i]
            next_row = register.iloc[i+1]
            if this_row[main_metric] is None or next_row[main_metric] is None:
                break
            diff = assess_differences(this_row['metrics'][metric], 
                                      next_row['metrics'][metric],
                                      trials=args.trials, 
                                      p_value=args.p_value)
            clusters.append(cluster_id)
            if diff:
                cluster_id += 1
        clusters.append(cluster_id)
        if len(clusters) < len(register):
            clusters += [cluster_id + 1] * (len(register) - len(clusters))
        register[f'cluster_{metric}'] = clusters

    # Save the results
    register = register.drop(columns=['metrics'])
    if args.append:
        register.to_csv(args.output, mode='a', header=False, index=False)
    else:
        register.to_csv(args.output, mode='w', header=True, index=False)
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
