# This will load a text file formatted as: sentance. sentance. sentance - sentance. ... etc.
# This creates two files: sources.txt and targets.txt
# formated as:
'''
source      target
s0          t0
s1          t1
'''
# NOTE: no whitespace in input.txt file other then single spaces or newlines!
# use regex to clean: replace (\n| {2,}) with single space " "

"""
    USAGE:
    seq2seq_dataset_creator.py -i input_file_path
    
    OPTIONAL OPTIONS:
    -d --> specify output folder, if this is not included it puts it in the same folder as the input file.
    -p --> specify prefix name on output files eg: 1234 = 1234_sources.txt
    -k --> change split type to count based, rather then NLP tokenize on sentence.
    -m --> adds a special character to signify end of sentence use like: -m "CHARACTER"
    
    EXAMPLES:
    like the demo code you provided, with end of sentence marker:
    python seq2seq_dataset_creator.py -i ../../data/aiethics/input.txt -k 25 -m "#"
    
    sources 1-10:
    The future of humanity is
     often viewed as a topic 
    for idle speculation.<>
    Yet our beliefs and assum
    ptions on this subject ma
    tter shape decisions in b
    oth our personal lives an
    d public policy  decision
    s that have very real and
     sometimes unfortunate co
    
    targets 1-10:
     often viewed as a topic 
    for idle speculation.<>
    Yet our beliefs and assum
    ptions on this subject ma
    tter shape decisions in b
    oth our personal lives an
    d public policy  decision
    s that have very real and
     sometimes unfortunate co
    nsequences.<>
"""

import os, argparse, re, nltk, codecs

from processor_chain import ProcessorChain
from preprocessors.sanitizer import Processor as Sanitizer

nltk.download("punkt")



def writeout(file_path, data):
    with codecs.open(file_path, "wd", 'utf-8') as sf:
        for line in data:
            sf.write(line + "\n")



def tokenize(input_file, count=None, mark=None):
    with codecs.open(input_file, "r", 'utf-8') as f:
        data = f.read()

        sources = []
        targets = []

        # preprocess
        chain = ProcessorChain( )
        regxp = "[^!' $\&\-,.;:\?ACBEDGFIHKJMLONQPSRUTWVYXZacbedgfihkjmlonqpsrutwvyxz]"
        chain.load(Sanitizer(regxp))
        processed_data = chain.run(data)

        # split on punctuation
        tokens = nltk.tokenize.sent_tokenize(processed_data)

        # mark sentence endings
        if mark:
            for idx,token in enumerate(tokens):
                tokens[idx] = token + mark

        # split by k factor
        if ( count ):
            # grab sections & re-tokenize by k.
            k = int(args.count)
            new_tokens = []
            for token in tokens:
                temp = [token[i:i+k] for i in range(0, len(token), k)]
                
                # fill sentence ends
                if mark: 
                    eol = temp[len(temp)-1]
                    while len(eol) < k:
                        eol += mark
                    temp[len(temp)-1] = eol

                new_tokens = new_tokens + temp
            tokens = new_tokens

        # make the sources and targets
        for idx,token in enumerate(tokens):
            if len(tokens) == idx+1:
                break # hacky way to stop this at the right point.
            sources.append(tokens[idx])
            targets.append(tokens[idx+1])

        return sources, targets



def create_outputs(args):
        # extract tokens
        sources,targets = tokenize(args.input, args.count, args.mark)

        # write out files to the right folder with the right name.
        prefix = ( args.prefix + "_" ) if args.prefix else ""
        dirname = args.directory if args.directory else os.path.dirname(args.input)
            
        sources_file = os.path.join(dirname, prefix + "sources.txt")
        targets_file = os.path.join(dirname, prefix + "targets.txt")

        writeout(sources_file, sources)
        writeout(targets_file, targets)






# run!!
parser = argparse.ArgumentParser(description='This will load a text file and split it into source/target for training based on the following parameters.')
parser.add_argument('-i','--input', help='Input file name',required=True)
parser.add_argument('-d','--directory',help='Output directory name for source/target.txt', required=False)
parser.add_argument('-p','--prefix',help='Prefix on output files source/target.txt', required=False)
parser.add_argument('-k','--count',help='Override to split on character count, rather then punctuation', required=False)
parser.add_argument('-m','--mark',help='Mark sentance endings with character -m CHAR', required=False)
args = parser.parse_args()

create_outputs(args)











