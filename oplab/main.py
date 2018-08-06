import argparse
import csv
import importlib
import json
import logging
import os
import sys
import time
from pathlib import Path

import texttable as tt
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from tklearn.metrics import pearson_corr, spearman_corr, optimize_threshold
from tklearn.utils import save_pipeline, load_pipeline
from tklearn.utils.collections import isiterable

from oplab.const import COMMAND_HELP, TRAIN_COMMAND_HELP

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


class TextkitLab:
    def __init__(self):
        self.working_dir = Path().absolute()
        logger.info('working directory is `{}`'.format(self.working_dir))
        parser = argparse.ArgumentParser(description='Executes OpinionLab commands.', usage=COMMAND_HELP)
        parser.add_argument('command', help='Sub-command to run')
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            logger.error('unrecognized command entered. Please see the documentation for more info.')
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()

    def train(self):
        parser = argparse.ArgumentParser(
            description='Trains models on the provided datasets using provided models and features.',
            usage=TRAIN_COMMAND_HELP)
        parser.add_argument('-c', '--config', help='path to config file.')
        parser.add_argument('-m', '--model', help='path to model output directory.')
        args = parser.parse_args(sys.argv[2:])
        config_path = os.path.join(self.working_dir, args.config)
        logger.info('processing configs: {}'.format(config_path))
        dataset, features, model, configs = _process_configs(config_path)
        module_name = 'oplab.datasets.{}'.format(dataset['name'])
        try:
            dataset, _ = importlib.import_module(module_name).read_dataset(**dataset)
            module_name = 'oplab.models.{}'.format(model['name'])
            # Building model <- Feature extraction and training
            model_builder = importlib.import_module(module_name)
            start = time.time()
            pipeline = model_builder.build_model(dataset, features, model)
            end = time.time()
            logger.info('model trained in {} sec'.format(end - start))
            model_path = os.path.join(self.working_dir, args.model)
            start = time.time()
            save_pipeline(model_path, pipeline)
            with open(os.path.join(model_path, 'configs.json'), 'w') as f:
                json.dump(configs, f)
            end = time.time()
            logger.info('pipeline saved in {} sec at {}'.format(end - start, model_path))
        except ModuleNotFoundError:
            logger.error('can\'t find module named `{}`'.format(module_name))
            exit(1)

    def t(self):
        return self.train

    def evaluate(self):
        parser = argparse.ArgumentParser(description='Predicts given instances and saves results..')
        parser.add_argument('-m', '--model', help='path to model folder.')
        parser.add_argument('-d', '--dataset', help='dataset to be used.')
        parser.add_argument('-e', '--metrics', nargs='*', help='metrics to be evaluated on.')
        parser.add_argument('-o', '--output', help='path to predictions file.')
        args = parser.parse_args(sys.argv[2:])
        model_path = os.path.join(self.working_dir, args.model)
        start = time.time()
        pipeline = load_pipeline(model_path)
        config_path = os.path.join(model_path, 'configs.json')
        dataset, features, model, configs = _process_configs(config_path)
        if args.dataset:
            dataset = {'name': args.dataset}
        end = time.time()
        logger.info('model loaded in {} seconds'.format(end - start, model_path))
        module_name = 'oplab.datasets.{}'.format(dataset['name'])
        try:
            dataset, categories = importlib.import_module(module_name).read_dataset(**dataset)
            text_lst = dataset[-1][0]
            y_true = dataset[-1][1]
            y_pred = pipeline.predict(text_lst)
            if categories is not None and len(y_true) > 0 and not isiterable(y_true[0]):
                true_labels = map(lambda x: categories[x], y_true)
                pred_labels = map(lambda x: categories[x], y_pred)
            else:
                true_labels = y_true
                pred_labels = y_pred
            if args.output:
                with open(os.path.join(self.working_dir, args.output + ('' if ''.endswith('.csv') else '.csv')), 'w',
                          encoding='utf-8') as f:
                    writer = csv.writer(f, lineterminator='\n')
                    writer.writerows(list(zip(text_lst, true_labels, pred_labels)))
            if args.metrics:
                _print_scores(y_true, y_pred, args.metrics)
        except ModuleNotFoundError:
            logger.error('can\'t find module named `{}`'.format(module_name))
            exit(1)

    def e(self):
        self.evaluate()


def _print_scores(y_true, y_pred, metrics):
    tab = tt.Texttable()
    print('\nTable: Evaluation Scores')
    headings = ['Metric', 'Score']
    tab.header(headings)
    for m in metrics:
        score = None
        try:
            if m == 'accuracy':
                y_pred = optimize_threshold(y_true, y_pred, accuracy_score)
                score = accuracy_score(y_true, y_pred)
            elif m == 'precision':
                y_pred = optimize_threshold(y_true, y_pred, precision_score)
                score = precision_score(y_true, y_pred)
            elif m == 'precision.micro':
                y_pred = optimize_threshold(y_true, y_pred, precision_score, average='micro')
                score = precision_score(y_true, y_pred, average='micro')
            elif m == 'precision.macro':
                y_pred = optimize_threshold(y_true, y_pred, precision_score, average='macro')
                score = precision_score(y_true, y_pred, average='macro')
            elif m == 'recall':
                score = recall_score(y_true, y_pred)
            elif m == 'recall.micro':
                y_pred = optimize_threshold(y_true, y_pred, recall_score, average='micro')
                score = recall_score(y_true, y_pred, average='micro')
            elif m == 'recall.macro':
                y_pred = optimize_threshold(y_true, y_pred, recall_score, average='macro')
                score = recall_score(y_true, y_pred, average='macro')
            elif m == 'f1':
                y_pred = optimize_threshold(y_true, y_pred, f1_score)
                score = f1_score(y_true, y_pred)
            elif m == 'f1.micro':
                y_pred = optimize_threshold(y_true, y_pred, f1_score, average='micro')
                score = f1_score(y_true, y_pred, average='micro')
            elif m == 'f1.macro':
                y_pred = optimize_threshold(y_true, y_pred, f1_score, average='macro')
                score = f1_score(y_true, y_pred, average='macro')
            elif m == 'pearson_corr':
                score = pearson_corr(y_true, y_pred)
            elif m == 'spearman_corr':
                score = spearman_corr(y_true, y_pred)
            tab.add_row([m, score])
        except Exception as e:
            tab.add_row([m, 'Invalid'])
    s = tab.draw()
    print(s)


def _process_configs(config_path):
    try:
        with open(config_path) as f:
            configs = json.load(f)
            try:
                return configs['dataset'], configs['features'], configs['model'], configs
            except KeyError:
                logger.error('Invalid config format. Please see documentation for more information.')
                exit(1)
    except FileNotFoundError:
        logger.error('cannot find the config file at `{}`.'.format(config_path))
        exit(1)


if __name__ == '__main__':
    TextkitLab()
