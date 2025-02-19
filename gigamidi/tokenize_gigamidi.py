"""Modified from train/tokenize-lakh.py, tokenizes a MIDI dataset"""
import istarmap_helper

import os
from argparse import ArgumentParser
from multiprocessing import Pool, RLock
from glob import glob

from tqdm import tqdm
import numpy as np

from anticipation import ops
from anticipation.config import *
from anticipation.vocab import *
from anticipation.tokenize import (
    maybe_tokenize,
    extract_spans,
    extract_random,
    extract_instruments,
    ANTICIPATION_RATES
)


# NOTE(Shih-Lun): 682 * 3 = 2046 --> aiming for 2K tokens per sequence
M = 682

# NOTE(Shih-Lun): chunks with <512 remaining AMT tokens are discarded
MIN_SEQLEN = 512

OUTPUT_SUBDIR = "amt-seq2k-autoregress"

def tokenize(datafile, output, augment_factor, idx=0, debug=False):
    # print("[datafile]", datafile)
    tokens = []
    all_truncations = 0
    seqcount = rest_count = 0
    stats = 4*[0] # (short, long, too many instruments, inexpressible)
    np.random.seed(0)

    assert output == OUTPUT_SUBDIR
    out_dir = os.path.dirname(datafile) + f"/{output}"
    os.makedirs(out_dir, exist_ok=True)
    # print(out_dir)


    with open(datafile, 'r') as f:
        all_events, truncations, status = maybe_tokenize([int(token) for token in f.read().split()])

    if status > 0: # this file is filtered out (not included)
        stats[status-1] += 1
        if debug:
            print(f"Discarded {datafile} due to status {status}")
        return (seqcount, rest_count, stats[0], stats[1], stats[2], stats[3], all_truncations)

    instruments = list(ops.get_instruments(all_events).keys())
    end_time = ops.max_time(all_events, seconds=False)

    # different random augmentations
    for k in range(augment_factor):
        concatenated_tokens = []

        if k % 10 == 0:
            # no augmentation
            events = all_events.copy()
            controls = []
        elif k % 10 == 1:
            # span augmentation
            lmbda = .05
            events, controls = extract_spans(all_events, lmbda)
        elif k % 10 < 6:
            # random augmentation
            r = np.random.randint(1,ANTICIPATION_RATES)
            events, controls = extract_random(all_events, r)
        else:
            if len(instruments) > 1:
                # instrument augmentation: at least one, but not all instruments
                u = 1+np.random.randint(len(instruments)-1)
                subset = np.random.choice(instruments, u, replace=False)
                events, controls = extract_instruments(all_events, subset)
            else:
                # no augmentation
                events = all_events.copy()
                controls = []

        if len(concatenated_tokens) == 0:
            z = ANTICIPATE if k % 10 != 0 else AUTOREGRESS

        all_truncations += truncations
        events = ops.pad(events, end_time)
        rest_count += sum(1 if tok == REST else 0 for tok in events[2::3])
        tokens, controls = ops.anticipate(events, controls)
        assert len(controls) == 0 # should have consumed all controls (because of padding)

        # NOTE(Shih-Lun): omitted for strictly separated pieces
        # tokens[0:0] = [SEPARATOR, SEPARATOR, SEPARATOR]

        concatenated_tokens.extend(tokens)
        orig_len = len(concatenated_tokens)

        # write out full sequences to file
        while len(concatenated_tokens) >= MIN_SEQLEN:
            # print(len(concatenated_tokens))
            seq = concatenated_tokens[0:EVENT_SIZE*M]
            concatenated_tokens = concatenated_tokens[EVENT_SIZE*M:]

            # relativize time to the context
            seq = ops.translate(seq, -ops.min_time(seq, seconds=False), seconds=False)
            assert ops.min_time(seq, seconds=False) == 0
            if ops.max_time(seq, seconds=False) >= MAX_TIME:
                stats[3] += 1
                continue

            # if seq contains SEPARATOR, global controls describe the first sequence
            seq.insert(0, z)

            with open(f"{out_dir}/segment-{seqcount + 1:03d}.txt", "w") as outfile:
                outfile.write(' '.join([str(tok) for tok in seq]) + '\n')

            # outfile.write(filename.split("midicaps/")[-1].split(".compound")[0] + " | ")
            # outfile.write(' '.join([str(tok) for tok in seq]) + '\n')
            seqcount += 1

        # grab the current augmentation controls if we didn't already
        z = ANTICIPATE if k % 10 != 0 else AUTOREGRESS


    if debug:
        fmt = 'Processed {} sequences (discarded {} tracks, discarded {} seqs, added {} rest tokens), Original Length: {}'
        print(fmt.format(seqcount, stats[0]+stats[1]+stats[2], stats[3], rest_count, orig_len))

    return (seqcount, rest_count, stats[0], stats[1], stats[2], stats[3], all_truncations)


def main(args):
    if args.interarrival:
        raise NotImplementedError('Interarrival-time encoding is not supported')
    
    encoding = 'interarrival' if args.interarrival else 'arrival'
    print('Tokenizing GigaMIDI')
    print(f'  encoding type: {encoding}')

    print('Tokenization parameters:')
    print(f'  anticipation interval = {DELTA}s')
    print(f'  augment = {args.augment}x')
    print(f'  max track length = {MAX_TRACK_TIME_IN_SECONDS}s')
    print(f'  min track length = {MIN_TRACK_TIME_IN_SECONDS}s')
    print(f'  min track events = {MIN_TRACK_EVENTS}')

    files = glob(os.path.join(args.datadir, f'**/*.amt-compound.txt'), recursive=True)
    print(f"  found {len(files)} files to tokenize")
#     exit()

#     outputs = [os.path.join(args.datadir, f'tokenized-events-{s}.txt') for s in LAKH_SPLITS]

    # don't augment the valid/test splits
    augment = [1 if "valid" in f or "test" in f else args.augment for f in files]

    # parallel tokenization drops the last chunk of < M tokens
    # if concerned about waste: process larger groups of datafiles
#     func = tokenize_ia if args.interarrival else tokenize
    func = tokenize

    seq_count = rest_count = too_short = too_long = too_manyinstr = discarded_seqs = truncations = 0

    pool_args = [(f, OUTPUT_SUBDIR, a, i) for f, a, i in zip(files, augment, range(len(files)))]
    with Pool(processes=PREPROC_WORKERS) as pool:
        for result in tqdm(pool.istarmap(func, pool_args), total=len(files)):
            seq_count += result[0]
            rest_count += result[1]
            too_short += result[2]
            too_long += result[3]
            too_manyinstr += result[4]
            discarded_seqs += result[5]
            truncations += result[6]

    rest_ratio = round(100*float(rest_count)/(seq_count*M),2)

    trunc_type = 'interarrival' if args.interarrival else 'duration'
    trunc_ratio = round(100*float(truncations)/(seq_count*M),2)

    print('Tokenization complete.')
    print(f'  => Processed {seq_count} training sequences')
    print(f'  => Inserted {rest_count} REST tokens ({rest_ratio}% of events)')
    print(f'  => Discarded {too_short+too_long+too_manyinstr} event sequences')
    print(f'      - {too_short} too short')
    print(f'      - {too_long} too long')
    print(f'      - {too_manyinstr} too many instruments')
    print(f'  => Discarded {discarded_seqs} training sequences')
    print(f'  => Truncated {truncations} {trunc_type} times ({trunc_ratio}% of {trunc_type}s)')

    print('Remember to shuffle the training split!')

if __name__ == '__main__':
    parser = ArgumentParser(description='tokenizes a MIDI dataset')
    parser.add_argument('datadir', help='directory containing preprocessed MIDI to tokenize')
    parser.add_argument('-k', '--augment', type=int, default=1,
            help='dataset augmentation factor (multiple of 10)')
    parser.add_argument('-i', '--interarrival',
            action='store_true',
            help='request interarrival-time enocoding (default to arrival-time encoding)')

    main(parser.parse_args())
