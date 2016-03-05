# show V's and X's on baseline output file

def main():
    predicted_file = '/Users/roeeaharoni/research_data/sigmorphon2016-master/src/baseline/baseline.german_task1.txt'
    gold_file = '/Users/roeeaharoni/research_data/sigmorphon2016-master/data/german-task1-dev'
    output_file = './baseline_error_analysis.txt'

    with open(predicted_file) as predicted:
        predicted_lines = predicted.readlines()
        with open(gold_file) as gold:
            gold_lines = gold.readlines()
            if not len(gold_lines) == len(predicted_lines):
                print 'file lengths mismatch, {0} lines in gold and {1} lines in prediction'.format(len(gold_lines),
                                                                                                    len(predicted_lines))
                return
            else:
                with open(output_file, 'w') as output:
                    morph2results = {}
                    for i, predicted_line in enumerate(predicted_lines):
                        output_line = ''
                        [lemma, morph, predicted_inflection] = predicted_line.split()
                        [lemma, morph, gold_inflection] = gold_lines[i].split()
                        output_line = lemma + '\t\t' + morph + '\t\t' + 'gold: ' + gold_inflection + '\t\tpredicted: ' + predicted_inflection
                        if predicted_inflection == gold_inflection:
                            #output_line += ' V\n'
                            continue
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