import time
import random
import argparse

import torch
import torch.nn as nn
from torch import optim

from nltk import bleu_score

from data_preprocess import Data_Preprocess
from encoder_rnn import Encoder_RNN
from decoder_rnn import Decoder_RNN
from train_network import Train_Network
from helper import Helper

use_cuda = torch.cuda.is_available()
helpFn = Helper()

file_name = ""
def print_file(s):
    file = open(file_name, "a")
    file.write(str(s) + '\n')
    file.close()


def train_iters(file_name, model, input_lang, output_lang, pairs, max_length, batch_size=1,
                n_iters=10, learning_rate=0.0005, tracking_pair=None, print_every=1, plot_every=1):

    start = time.time()
    start2 = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_trainable_parameters = list(filter(lambda p: p.requires_grad, model.encoder.parameters()))
    decoder_trainable_parameters = list(filter(lambda p: p.requires_grad, model.decoder.parameters()))

    encoder_optimizer = optim.RMSprop(encoder_trainable_parameters, lr=learning_rate)
    decoder_optimizer = optim.RMSprop(decoder_trainable_parameters, lr=learning_rate)

    ''' Lists that will contain data in the form of tensors. '''
    in_seq = []
    out_seq = []
    input_lengths = []

    samples = 0
    ''' Get all data points '''
    for samples, pair in enumerate(pairs):
        input_variable, input_length, target_variable, _ = helpFn.variables_from_pair(input_lang, output_lang, pair)

        in_seq.append(input_variable)
        out_seq.append(target_variable)
        input_lengths.append(input_length)

    samples -= (samples + 1) % batch_size
    criterion = nn.NLLLoss(ignore_index=0)

    print('Beginning Model Training.')

    for epoch in range(1, n_iters + 1):
        for i in range(0, samples, batch_size):
            input_variables = in_seq[i : i + batch_size] # Batch Size x Sequence Length
            target_variables = out_seq[i : i + batch_size]
            lengths = input_lengths[i : i + batch_size]

            loss = model.train(input_variables, target_variables, lengths,
                               encoder_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss
            plot_loss_total += loss

        if tracking_pair:
            evaluate_specific(model, input_lang, tracking_pair)

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_file("Epoch" + str(epoch) + " loss: " + str(print_loss_avg))
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (helpFn.time_slice(start, epoch / n_iters),
                                         epoch, epoch / n_iters * 100, print_loss_avg))

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

        if epoch % 4 == 0:
            learning_rate = max(learning_rate/2.0, 0.000005)
            encoder_optimizer = optim.RMSprop(encoder_trainable_parameters, lr=learning_rate)
            decoder_optimizer = optim.RMSprop(decoder_trainable_parameters, lr=learning_rate)

        print_file("Epoch " + str(epoch) + " time/epoch " + str(time.time() - start2))
        start2 = time.time()
        print("\n")

    helpFn.show_plot(plot_losses)

def evaluate(train_network, input_lang, sentence):
    input_variable, _ = helpFn.variable_from_sentence(input_lang, sentence)
    output_words, attentions = train_network.evaluate([input_variable])
    return output_words, attentions

def evaluate_specific(train_network, input_lang, pair, name='tracking_pair'):
    print('>', pair[0])
    print('=', pair[1])
    output_words, attentions = evaluate(train_network, input_lang, pair[0])
    output_sentence = ' '.join(output_words)
    print('<', output_sentence)
    print('BLEU Score', bleu_score.corpus_bleu([output_sentence], [pair[1]]))
    helpFn.show_attention(pair[0], output_words, attentions, name=name)

def evaluate_randomly(train_network, input_lang, pairs, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        evaluate_specific(train_network, input_lang, pair, name=str(i))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-z", "--hidden_size", type=int, help="LSTM Embedding Size", default=256)
    parser.add_argument("-b", "--batch_size", type=int, help="Batch Size", default=32)
    parser.add_argument("-l", "--max_length", type=int, help="Maximum Sentence Length.", default=20)
    parser.add_argument("--num_layers", type=int, help="Number of layers in Encoder and Decoder", default=2)
    parser.add_argument("-e", "--embedding_size", type=int, help="Length of the embedding vector.", default=300)
    parser.add_argument("-v", "--vocab_size", type=int, help="Maximum number of words in the vocab.", default=8000)
    parser.add_argument("-n", "--num_iters", type=int, help="Number of epochs", default=20)
    parser.add_argument("-t", "--target_term_id", type=str, help="Target term id.", default='7')
    parser.add_argument("-d", "--dataset", type=str, help="Dataset file.", default='../../Drive/Information Extraction Team/Dataset/final.tsv')

    args = parser.parse_args()

    hidden_size = args.hidden_size
    num_iters = args.num_iters
    batch_size = args.batch_size
    max_length = args.max_length
    num_layers = args.num_layers
    embedding_size = args.embedding_size
    vocab_size = args.vocab_size

    print('Model Parameters:')
    print('Hidden Size          :', hidden_size)
    print('Batch Size           :', batch_size)
    print('Number of Layers     :', num_layers)
    print('Embedding Size       :', embedding_size)
    print('Max. input length    :', max_length)
    print('Max. vocab size      :', vocab_size)
    print('Num epochs           :', num_iters)
    print('--------------------------------------\n')

    file_name = "./logs/" + str(num_iters) + "_" + str(vocab_size) + "_" + str(hidden_size) + ".txt"
    print_file('Model Parameters:\n')
    print_file('Hidden Size          :' + str(hidden_size))
    print_file('Batch Size           :' + str(batch_size))
    print_file('Number of Layers     :' + str(num_layers))
    print_file('Embedding Size       :' + str(embedding_size))
    print_file('Max. input length    :' + str(max_length))
    print_file('Max. vocab size      :' + str(vocab_size))
    print_file('Num epochs           :' + str(num_iters))
    print_file('--------------------------------------')

    data_preprocess = Data_Preprocess(max_length, vocab_size, vocab_size)
    input_lang, output_lang, pairs = data_preprocess.prepare_data()
    tracking_pair = random.choice(pairs)
    print(tracking_pair)

    ''' Generate and learn embeddings '''
    print("input_l")
    encoder = Encoder_RNN(hidden_size, (len(input_lang.word2index), embedding_size), batch_size=batch_size, num_layers=num_layers)
    decoder = Decoder_RNN(hidden_size, (len(output_lang.word2index), embedding_size), num_layers=num_layers)

    if use_cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    print("Training Network.")
    train_network = Train_Network(encoder, decoder, output_lang, max_length, batch_size=batch_size, num_layers=num_layers)
    train_iters(file_name, train_network, input_lang, output_lang, pairs, max_length, batch_size=batch_size, tracking_pair=tracking_pair, n_iters = num_iters)

    evaluate_randomly(train_network, input_lang, pairs)
