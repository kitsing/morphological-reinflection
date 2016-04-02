# show V's and X's on predicted output file according to gold file

def main():

    evaluate('/Users/roeeaharoni/GitHub/morphological-reinflection/results/joint_structured_arabic_results.txt.best.predictions',
             '/Users/roeeaharoni/research_data/sigmorphon2016-master/data/arabic-task1-dev',
             '/Users/roeeaharoni/GitHub/morphological-reinflection/results/joint_structured_arabic_results.txt.best.predictions.error_analysis')
    return

    predicted_file_format = '/Users/roeeaharoni/GitHub/morphological-reinflection/results/\
joint_structured_{0}_results.txt.best.predictions'

    gold_file_format = '/Users/roeeaharoni/research_data/sigmorphon2016-master/data/{0}-task1-dev'

    output_file_format = '/Users/roeeaharoni/GitHub/morphological-reinflection/results/\
joint_structured_{0}_results.txt.best.predictions_error_analysis.txt'

    langs = ['arabic', 'finnish', 'georgian', 'russian', 'german', 'turkish', 'spanish', 'navajo']

    for lang in langs:
        predicted_file = predicted_file_format.format(lang)
        gold_file = gold_file_format.format(lang)
        output_file = output_file_format.format(lang)
        evaluate(predicted_file, gold_file, output_file)
        print 'created error analysis for {0} in: {1}'.format(lang, output_file)


def evaluate(predicted_file, gold_file, output_file):
    with open(predicted_file) as predicted:
        predicted_lines = predicted.readlines()
        with open(gold_file) as gold:
            gold_lines = gold.readlines()
            if not len(gold_lines) == len(predicted_lines):
                print 'file lengths mismatch, {0} lines in gold and {1} lines in prediction'.format(len(gold_lines),
                                                                                                    len(predicted_lines))

            else:
                with open(output_file, 'w') as output:
                    morph2results = {}
                    for i, predicted_line in enumerate(predicted_lines):
                        output_line = ''
                        [pred_lemma, pred_morph, predicted_inflection] = predicted_line.split()
                        [lemma, morph, gold_inflection] = gold_lines[i].split()

                        if pred_lemma != lemma:
                            print 'mismatch in index' + str(i)
                            return

                        output_line = lemma + '\t\t' + morph + '\t\t' + 'gold: ' + gold_inflection + '\t\tpredicted: ' + predicted_inflection
                        if predicted_inflection == gold_inflection:
                            output_line += ' V\n'
                        else:
                            output_line += ' X\n'

                        if morph in morph2results:
                            morph2results[morph].append(output_line)
                        else:
                            morph2results[morph] = [output_line]
                        # output.write(output_line)

                    for morph in morph2results:
                        output.write('\n\n#################################\n\n')
                        for line in morph2results[morph]:
                            output.write(line)

if __name__ == '__main__':
    main()