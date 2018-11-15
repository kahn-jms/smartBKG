#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# All PDG codes known to man
# 0 at the bottom if for the motherPDG code given to the Y4S by CCNPandasPickle.py
# James Kahn

pdgTokens = [
    '0',
    '(-->',
    ' <--)',
    '329',
    '110553',
    '5',
    '-12',
    '425',
    '553',
    '-20513',
    '-311',
    '13122',
    '323',
    '-11',
    '510',
    '-325',
    '20513',
    '-321',
    '110555',
    '100313',
    '10533',
    '310',
    '130553',
    '-10521',
    '20423',
    '521',
    '10521',
    '5114',
    '-13122',
    '-10533',
    '150',
    '-10511',
    '-13212',
    '5232',
    '-415',
    '-10213',
    '313',
    '-100313',
    '-23122',
    '20523',
    '15',
    '20333',
    '-100411',
    '-435',
    '2112',
    '-3314',
    '-413',
    '20213',
    '413',
    '82',
    '-2203',
    '-4',
    '9000553',
    '4422',
    '-541',
    '-3114',
    '-87',
    '30553',
    '100441',
    '-2214',
    '-10421',
    '-18',
    '13126',
    '3122',
    '2203',
    '9010443',
    '5322',
    '87',
    '3334',
    '10513',
    '-2103',
    '86',
    '-86',
    '-3212',
    '-20523',
    '14122',
    '-3126',
    '4432',
    '-2224',
    '515',
    '20313',
    '9000221',
    '-4424',
    '-10423',
    '-14122',
    '-10523',
    '-4412',
    '-431',
    '-4122',
    '4412',
    '10523',
    '-10411',
    '-4422',
    '-5114',
    '200555',
    '-30323',
    '10022',
    '-329',
    '130',
    '200551',
    '30323',
    '43122',
    '-43122',
    '10313',
    '-100413',
    '98',
    '-545',
    '10311',
    '-10431',
    '210553',
    '4122',
    '4132',
    '10431',
    '88',
    '-20323',
    '9910445',
    '321',
    '-4414',
    '3',
    '100413',
    '433',
    '-4334',
    '10411',
    '-2212',
    '-5103',
    '10553',
    '-13126',
    '-2',
    '-13',
    '5401',
    '1000020040',
    '3114',
    '30313',
    '-523',
    '83',
    '10113',
    '411',
    '85',
    '-5324',
    '220553',
    '100423',
    '-423',
    '53122',
    '113',
    '-513',
    '-211',
    '-20533',
    '10223',
    '480000000',
    '93',
    '20022',
    '-3214',
    '20113',
    '37',
    '3214',
    '4414',
    '335',
    '5132',
    '-319',
    '100213',
    '441',
    '10421',
    '5203',
    '-3203',
    '315',
    '-4332',
    '-20433',
    '-1000010020',
    '4103',
    '10543',
    '435',
    '5332',
    '-10541',
    '300553',
    '5224',
    '-2114',
    '3101',
    '-3101',
    '20433',
    '100553',
    '331',
    '-3216',
    '10423',
    '5214',
    '-323',
    '-14',
    '5222',
    '-315',
    '20443',
    '20413',
    '-5132',
    '-30363',
    '-411',
    '513',
    '44',
    '20533',
    '-3322',
    '3324',
    '-85',
    '97',
    '1',
    '9020443',
    '10511',
    '10333',
    '-4232',
    '-5',
    '4201',
    '-10323',
    '20223',
    '100223',
    '3224',
    '5201',
    '-20213',
    '30221',
    '-30313',
    '-30213',
    '325',
    '-10433',
    '10433',
    '20323',
    '-5332',
    '10213',
    '-213',
    '-543',
    '200553',
    '523',
    '13212',
    '-4201',
    '3203',
    '225',
    '-84',
    '-215',
    '84',
    '-4103',
    '7',
    '-5224',
    '-6',
    '6',
    '211',
    '-13124',
    '2114',
    '-521',
    '2103',
    '33',
    '-3303',
    '-4101',
    '-4301',
    '431',
    '-425',
    '17',
    '223',
    '11',
    '115',
    '8',
    '4301',
    '100211',
    '4203',
    '5212',
    '-3201',
    '-20423',
    '-4212',
    '120553',
    '5101',
    '4434',
    '-33122',
    '12',
    '94',
    '3201',
    '5334',
    '100551',
    '10443',
    '24',
    '215',
    '-10321',
    '557',
    '10555',
    '91',
    '423',
    '4114',
    '-327',
    '-82',
    '-4114',
    '-10313',
    '30223',
    '-8',
    '5122',
    '210551',
    '-3334',
    '5303',
    '1000010030',
    '33122',
    '-4403',
    '-13214',
    '531',
    '36',
    '-4124',
    '96',
    '100555',
    '95',
    '-20543',
    '92',
    '100221',
    '4312',
    '4403',
    '-4132',
    '-100421',
    '9000111',
    '2214',
    '3103',
    '-100211',
    '5503',
    '327',
    '-5303',
    '4303',
    '-5203',
    '100333',
    '4222',
    '30363',
    '-4434',
    '-3103',
    '34',
    '13214',
    '5103',
    '-511',
    '333',
    '511',
    '90',
    '4314',
    '-4324',
    '23122',
    '-3124',
    '-533',
    '25',
    '100421',
    '4424',
    '-41',
    '30443',
    '-525',
    '-3324',
    '213',
    '100323',
    '41',
    '30353',
    '-30353',
    '-3112',
    '4124',
    '4322',
    '-317',
    '3124',
    '-10531',
    '3216',
    '-15',
    '-3222',
    '319',
    '-20413',
    '525',
    '-1000020030',
    '-20313',
    '-9042413',
    '415',
    '23',
    '3322',
    '445',
    '10531',
    '-5334',
    '30343',
    '5403',
    '-5403',
    '4324',
    '-5222',
    '5301',
    '20553',
    '4332',
    '110551',
    '-421',
    '-30343',
    '-4203',
    '311',
    '100443',
    '-515',
    '13',
    '-5312',
    '-1000010030',
    '2',
    '5324',
    '2101',
    '-5401',
    '317',
    '-2101',
    '30213',
    '-4432',
    '3222',
    '-4112',
    '4212',
    '1000020030',
    '-3122',
    '421',
    '20555',
    '9000443',
    '350',
    '5312',
    '-4322',
    '-5201',
    '-44',
    '4112',
    '-100323',
    '99',
    '35',
    '-5301',
    '3112',
    '-5214',
    '-10513',
    '-3',
    '-5322',
    '10551',
    '10413',
    '443',
    '221',
    '81',
    '-24',
    '14',
    '2212',
    '-3224',
    '4334',
    '5314',
    '111',
    '-433',
    '9000211',
    '18',
    '4232',
    '10321',
    '-100213',
    '551',
    '3303',
    '-1',
    '-5314',
    '-313',
    '-531',
    '1114',
    '-2112',
    '-34',
    '10541',
    '535',
    '9920443',
    '-4222',
    '-535',
    '2224',
    '120555',
    '10441',
    '43',
    '-4312',
    '555',
    '20543',
    '1000010020',
    '-53122',
    '543',
    '545',
    '3314',
    '9030221',
    '-1114',
    '3212',
    '9042413',
    '541',
    '533',
    '100113',
    '-1000020040',
    '1103',
    '3312',
    '3126',
    '-5212',
    '-5112',
    '4101',
    '-4303',
    '10323',
    '-37',
    '-10311',
    '-4314',
    '-3312',
    '-100423',
    '-1103',
    '-7',
    '-9000211',
    '-16',
    '32',
    '21',
    '16',
    '4',
    '100557',
    '13124',
    '9010221',
    '100111',
    '4214',
    '23212',
    '-10413',
    '-5101',
    '-10543',
    '100411',
    '-4214',
    '-5122',
    '30113',
    '-5503',
    '5112',
    '-23212',
    '-4224',
    '4224',
    '530',
    '-17',
    '-5232',
    '22',
    '30643',
    '-30643',
    '30653',
    '-30653',
    '9000113',
    '9000213',
    '9020221',
    '10111',
    '10211',
    '100331',
    '9010113',
    '9010213',
    '10225',
    '227',
    '10115',
    '10215',
    '117',
    '217',
    '10331',
    '9010111',
    '9010211',
    '337',
    '9050225',
    '9060225',
    '119',
    '219',
    '229',
    '9080225',
    '9090225',
    '10315',
    '10325',
    '20315',
    '20325',
]
