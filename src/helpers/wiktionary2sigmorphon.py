import codecs
import csv
import sys

def unicode_csv_reader(unicode_csv_data, dialect=csv.excel, **kwargs):
    # csv.py doesn't do Unicode; encode temporarily as UTF-8:
    csv_reader = csv.reader(utf_8_encoder(unicode_csv_data),
                            dialect=dialect, **kwargs)
    for row in csv_reader:
        # decode UTF-8 back to Unicode, cell by cell:
        yield [unicode(cell, 'utf-8') for cell in row]

def utf_8_encoder(unicode_csv_data):
    for line in unicode_csv_data:
        yield line.encode('utf-8')

def main():

    inflections_path = '/Users/roeeaharoni/research_data/morphology/wiktionary-morphology-1.1/inflections_de_noun.csv'
    train_lemma_path = '/Users/roeeaharoni/research_data/morphology/wiktionary-morphology-1.1/base_forms_de_noun_train.txt'
    dev_lemma_path = '/Users/roeeaharoni/research_data/morphology/wiktionary-morphology-1.1/base_forms_de_noun_dev.txt'
    test_lemma_path = '/Users/roeeaharoni/research_data/morphology/wiktionary-morphology-1.1/base_forms_de_noun_test.txt'

    suffix = '.sigmorphon_format.txt'
    train_output_path = train_lemma_path + suffix
    dev_output_path = dev_lemma_path + suffix
    test_output_path = test_lemma_path + suffix

    # open train lemma file
    with codecs.open(train_lemma_path, encoding='utf8') as f:
        train_lemmas = [line.replace('\n','') for line in f]

    # open dev lemma file
    with codecs.open(dev_lemma_path, encoding='utf8') as f:
        dev_lemmas = [line.replace('\n','') for line in f]

    # open test lemma file
    with codecs.open(test_lemma_path, encoding='utf8') as f:
        test_lemmas = [line.replace('\n','') for line in f]

    # open inflections file
    with codecs.open(inflections_path, encoding='utf8') as f:
        reader = unicode_csv_reader(f)
        inflections = list(reader)

    # write new files

    # train
    print_matching_inflections_to_sigmorphon_file(inflections, train_lemmas, train_output_path)

    # dev
    print_matching_inflections_to_sigmorphon_file(inflections, dev_lemmas, dev_output_path)

    # test
    print_matching_inflections_to_sigmorphon_file(inflections, test_lemmas, test_output_path)


def print_matching_inflections_to_sigmorphon_file(inflections, base_forms, output_path):

    # sort inflections by base forms
    base2inflections = {}
    print len(inflections)

    for inflection in inflections:
        base_form = inflection[1]
        if base_form in base2inflections:
            base2inflections[base_form].append(inflection)
        else:
            base2inflections[base_form] = []
            base2inflections[base_form].append(inflection)

    with codecs.open(output_path, "w", encoding='utf8') as output_file:

        # read base forms
        for base in base_forms:

            if not base in base2inflections:
                print base + 'is not found in inflection file'
                continue

            # find their inflections
            for inflection in base2inflections[base]:

                # print their inflections to file
                # from:
                # Ubungen	Ubung	case=accusative:number=plural
                # to:
                # Ubung	case=accusative,number=plural Ubungen
                output_file.write(inflection[1] + '\t' + inflection[2].replace(':',',') + '\t' + inflection[0] + '\n')
    return

if __name__ == '__main__':
    main()