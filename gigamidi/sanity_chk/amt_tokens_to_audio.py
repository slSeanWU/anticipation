import glob
import json
import os
import random
import shutil
from multiprocessing import Pool

from argparse import ArgumentParser

from anticipation.synthesize import synthesize_midi
from anticipation.convert import events_to_midi
from anticipation.config import PREPROC_WORKERS
from anticipation.vocab import AUTOREGRESS, ANTICIPATE

def get_gigamidi_md5(token_file):
    return json.load(
       open(
           os.path.dirname(os.path.dirname(token_file)) + "/metadata.json"
       )
    )["md5"]

def main(args):
    all_token_files = glob.glob(os.path.join(args.datadir, "**/segment*.txt"), recursive=True)
    print(len(all_token_files))

    all_token_files = random.sample(all_token_files, args.nsamp)

    if args.remove_existing_outputs:
        shutil.rmtree(args.outputdir, ignore_errors=True)

    if not os.path.exists(args.outputdir):
        os.makedirs(args.outputdir)

    all_midi_paths = []

    for token_file in all_token_files:
        # print(token_file)
        md5 = get_gigamidi_md5(token_file)
        seg_num = int(os.path.basename(token_file).split("-")[1].split(".")[0])

        out_midi_path = os.path.join(args.outputdir, f"{md5}-segment{seg_num:03d}.mid")
        midi_obj = events_to_midi(
            [
                int(t) for t in open(token_file).read().split() 
                if int(t) not in {AUTOREGRESS, ANTICIPATE}
            ]
        )
        midi_obj.save(out_midi_path)

        all_midi_paths.append((out_midi_path,))

    with Pool(processes=PREPROC_WORKERS) as pool:
        pool.starmap(synthesize_midi, all_midi_paths)
        

if __name__ == "__main__":
    parser = ArgumentParser(
        description="sanity check if tokenized MIDI can be converted to reasonable audio"
    )
    parser.add_argument("datadir", help="directory containing AMT tokens (of MIDI files)")
    parser.add_argument("outputdir", help="directory to save converted MIDIs & synthesized audio")
    parser.add_argument("--nsamp", type=int, help="number of samples to convert", default=30)
    parser.add_argument("--remove_existing_outputs", type=int, help="remove contents in `outputdir`", default=1)

    main(parser.parse_args())