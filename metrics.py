import os
import sys
import torch
import sacrebleu
import fastwer
import subprocess
import tempfile
from time import time
from re import sub, findall
from shapely.geometry import Polygon
from unicodedata import normalize
from art import aggregators, scores, significance_tests
from cleanfid import fid
from skimage.metrics import structural_similarity as ssim
from PIL import Image

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
    return 100 * edits / lengths

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
        glob_ref_wl += len(sent_y.split())
    return 100 * global_scr / glob_ref_wl

def get_iou(x, y):
    '''
    Compute the IOU score between two lists of bounding boxes.
    Args:
        x: list of translated bounding boxes
        y: list of reference bounding boxes
    Returns:
        iou: IOU score
    '''
    ious = []
    for box_x, box_y in zip(x, y[0]):
        poly_x = Polygon(box_x)
        poly_y = Polygon(box_y)
        intersection = poly_x.intersection(poly_y).area
        union = poly_x.union(poly_y).area
        iou = intersection / union if union != 0 else 0
        ious.append(iou)
    return sum(ious) / len(ious)

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
        sys.stderr.write('Error: Beer not installed, please install it using setup.sh.\n')
        sys.exit(-1)

    # Delete aux files
    process = subprocess.Popen(('rm ' + hyps_file + ' '
                                + ' '.join(refs_files)).split(), stdout=subprocess.PIPE)
    beer = beer.decode('utf-8')
    beer = [float(s) for s in findall(r"score is ([0-9.]+)", beer)]
    return  sum(beer) / len(beer), beer

def get_fid(x, y):
    '''
    Compute the FID score between two lists of images.
    Args:
        x: list of generated images (numpy arrays)
        y: list of reference images (numpy arrays)
    Returns:
        fid: FID score
    '''
    with tempfile.TemporaryDirectory() as src_dir, \
         tempfile.TemporaryDirectory() as ref_dir:
        for i,img in enumerate(x):
            Image.fromarray(img).save(os.path.join(src_dir, f"image{i}.png"))
        for i,img in enumerate(y[0]):
            Image.fromarray(img).save(os.path.join(ref_dir, f"image{i}.png"))
        
        return fid.compute_fid(src_dir, 
                               ref_dir, 
                               mode="clean", 
                               device="cuda" if torch.cuda.is_available() else "cpu")

def get_ssim(x, y):
    '''
    Compute the SSIM score between two lists of images.
    Args:
        x: list of generated images (numpy arrays)
        y: list of reference images (numpy arrays)
    Returns:
        ssim_score: SSIM score
    '''
    ssim_scores = []
    y = y[0]
    for img_x, img_y in zip(x, y):
        score = ssim(img_x, img_y, channel_axis=2, data_range=img_y.max() - img_y.min())
        ssim_scores.append(score)
    return sum(ssim_scores) / len(ssim_scores)

def save_to_file(sentences):
    file = '/tmp/' + str(time()) + '_archer'
    with open(file, 'w') as f:
        f.write('\n'.join(sentences))
    return file

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

METRICS = {
    'bleu': get_bleu,
    'ter': get_ter,
    'chrf': get_chrf,
    'wer': get_wer,
    'bwer': get_bwer,
    'beer': get_beer,
    'fid': get_fid,
    'ssim': get_ssim,
    'iou': get_iou
}