with open('./db_raw.csv', 'r', encoding='utf-8') as f:
    text = f.read()
text = text.replace('ИЛЭ', '1').replace('ЛИИ', '2').replace('СЭИ', '3').replace('ЭСЭ', '4')
text = text.replace('СЛЭ', '5').replace('ЛСИ', '6').replace('ИЭИ', '7').replace('ЭИЭ', '8')
text = text.replace('СЭЭ', '9').replace('ЭСИ', '10').replace('ИЛИ', '11').replace('ЛИЭ', '12')
text = text.replace('ИЭЭ', '13').replace('ЭИИ', '14').replace('СЛИ', '15').replace('ЛСЭ', '16')
print(text, file=open('./_db_raw.csv', 'w', encoding='utf-8'))
