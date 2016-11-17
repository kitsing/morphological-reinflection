# morphological-reinflection

Usage:

    hard_attention.py [--cnn-mem MEM][--input=INPUT] [--hidden=HIDDEN] [--feat-input=FEAT] [--epochs=EPOCHS] [--layers=LAYERS] [--optimization=OPTIMIZATION] [--reg=REGULARIZATION][--learning=LEARNING] [--plot] [--eval] [--ensemble=ENSEMBLE] TRAIN_PATH DEV_PATH TEST_PATH RESULTS_PATH SIGMORPHON_PATH...

Arguments:
* TRAIN_PATH    destination path
* DEV_PATH      development set path
* TEST_PATH     test path
* RESULTS_PATH  results file to be written
* SIGMORPHON_PATH   sigmorphon root containing data, src dirs

Options:
* -h --help                     show this help message and exit
* --cnn-mem MEM                 allocates MEM bytes for (py)cnn
* --input=INPUT                 input vector dimensions
* --hidden=HIDDEN               hidden layer dimensions
* --feat-input=FEAT             feature input vector dimension
* --epochs=EPOCHS               amount of training epochs
* --layers=LAYERS               amount of layers in lstm network
* --optimization=OPTIMIZATION   chosen optimization method ADAM/SGD/ADAGRAD/MOMENTUM/ADADELTA
* --reg=REGULARIZATION          regularization parameter for optimization
* --learning=LEARNING           learning rate parameter for optimization
* --plot                        draw a learning curve plot while training each model
* --eval                        run evaluation without training
* --ensemble=ENSEMBLE           ensemble model paths, separated by comma

For example:

    python hard_attention.py --cnn-mem 4096 --input=100 --hidden=100 --feat-input=20 --epochs=100 --layers=2 --optimization=ADADELTA  /Users/roeeaharoni/research_data/sigmorphon2016-master/data/navajo-task1-train /Users/roeeaharoni/research_data/sigmorphon2016-master/data/navajo-task1-dev /Users/roeeaharoni/research_data/sigmorphon2016-master/data/navajo-task1-test /Users/roeeaharoni/Dropbox/phd/research/morphology/inflection_generation/results/navajo_results.txt /Users/roeeaharoni/research_data/sigmorphon2016-master/
