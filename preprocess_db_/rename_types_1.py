with open('./db_rus.csv', 'r', encoding='utf-8') as f:
    text = f.read()
text = text.replace('ИЛЭ', '1').replace('ЛИИ', '2').replace('СЭИ', '3').replace('ЭСЭ', '4')
text = text.replace('СЛЭ', '5').replace('ЛСИ', '6').replace('ИЭИ', '7').replace('ЭИЭ', '8')
text = text.replace('СЭЭ', '9').replace('ЭСИ', '10').replace('ИЛИ', '11').replace('ЛИЭ', '12')
text = text.replace('ИЭЭ', '13').replace('ЭИИ', '14').replace('СЛИ', '15').replace('ЛСЭ', '16')
print(text, file=open('./_db_rus.csv', 'w', encoding='utf-8'))


with open('./db_eng_1.csv', 'r', encoding='utf-8') as f:
    text = f.read()
text = text.replace('ILE', '1').replace('LII', '2').replace('SEI', '3').replace('ESE', '4')
text = text.replace('SLE', '5').replace('LSI', '6').replace('IEI', '7').replace('EIE', '8')
text = text.replace('SEE', '9').replace('ESI', '10').replace('ILI', '11').replace('LIE', '12')
text = text.replace('IEE', '13').replace('EII', '14').replace('SLI', '15').replace('LSE', '16')
print(text, file=open('./_db_eng_1.csv', 'w', encoding='utf-8'))


with open('./db_eng_2.csv', 'r', encoding='utf-8') as f:
    text = f.read()
text = text.replace('ILE', '1').replace('LII', '2').replace('SEI', '3').replace('ESE', '4')
text = text.replace('SLE', '5').replace('LSI', '6').replace('IEI', '7').replace('EIE', '8')
text = text.replace('SEE', '9').replace('ESI', '10').replace('ILI', '11').replace('LIE', '12')
text = text.replace('IEE', '13').replace('EII', '14').replace('SLI', '15').replace('LSE', '16')
print(text, file=open('./_db_eng_2.csv', 'w', encoding='utf-8'))


with open('./db_eng_1b.csv', 'r', encoding='utf-8') as f:
    text = f.read()
text = text.replace('ILE', '1').replace('LII', '2').replace('SEI', '3').replace('ESE', '4')
text = text.replace('SLE', '5').replace('LSI', '6').replace('IEI', '7').replace('EIE', '8')
text = text.replace('SEE', '9').replace('ESI', '10').replace('ILI', '11').replace('LIE', '12')
text = text.replace('IEE', '13').replace('EII', '14').replace('SLI', '15').replace('LSE', '16')
print(text, file=open('./_db_eng_1b.csv', 'w', encoding='utf-8'))


with open('./db_eng_2b.csv', 'r', encoding='utf-8') as f:
    text = f.read()
text = text.replace('ILE', '1').replace('LII', '2').replace('SEI', '3').replace('ESE', '4')
text = text.replace('SLE', '5').replace('LSI', '6').replace('IEI', '7').replace('EIE', '8')
text = text.replace('SEE', '9').replace('ESI', '10').replace('ILI', '11').replace('LIE', '12')
text = text.replace('IEE', '13').replace('EII', '14').replace('SLI', '15').replace('LSE', '16')
print(text, file=open('./_db_eng_2b.csv', 'w', encoding='utf-8'))
