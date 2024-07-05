from MustRunBeforeUsingHanSpellLibrary import correct_passportKey
correct_passportKey()

from hanspell import spell_checker as sc
from pykospacing import Spacing

while(True):
    sentence = input('입력: ')

    spacing = Spacing(rules=[''])
    spacing.set_rules_by_csv('dict.cvs', 'word')

    spacingSent = spacing(sentence)
    spelledSent = sc.check(sentence)
    print('----- PyKoSpacing(띄어쓰기) -----')
    print(spacingSent)
    print()
    print('----- Py-Hanspell(철자/띄어쓰기) -----')
    print(spelledSent.checked)
    print()
    print('dict 형식:')
    print(spelledSent)
    if ( input("종료(q):") != 'q'): continue
    break
