Описание данных и парсингового кода
===================================

Данные цен финансовых активов
-----------------------------
### Все данные ниже представляют собой данные для целевой переменной - сумма прибыли/убытка по опционной стратегии за период бэктестинга. Целевая переменная - это набор точек прибыли за каждый интервал времени (допустим, прибыль за 5 дней).

Filler пустых значений везде = пустое значение (str = '')

1. **bbg/*.xlsx**\
    (timestamp, price)
    > Дневные данные Блумберга по EURUSD и USDCNH (только оффшорные, так как нормальной ликвидности во всех рублевых парах там нет с марта).
***
2. **bbg/*.csv**\
   (timestamp, price)
   > Cгенерированный "синтетический курс" (см. пункт 5).
***
3. **forts/*.csv**\
   (option code, option type: binary = C => Call or P => Put, id deal, option price, deal volume in № options, deal volume in RUB, deal side: binary = B => Buy or S => Sell)
   > Опционные данные с Мосбиржи.
***
4. **loki/*.csv**\
   (timestamp, price)
   > Часовые данные с MOEX. (Не основной вариант - лучше использовать пункт №5)
***
5. **(!) moex/*.csv**\
   (timestamp, bid price, ask price)
   > Самые полезные данные спотовые, часовые данные с MOEX (полный датасет в отличие от 4).
***
6. **moex_fixing/*.xml**\ и *.csv (идентичные)"
   (timestamp, price)
   > Дневные спотовые данные из открытого доступа.
***
12. **rbi**\
   (timestamp, bid price, ask price)
   > Часовые спотовые данные оффшороного FX.
***

Данные аналитических и новостных текстов
----------------------------------------
### Все данные ниже представляют собой признаки объектов.


Парсинг и обработка данных
--------------------------
1. **src/working_with_files/backtester.py**\
   > Класс Backtester - процесс, проводящий бэктестирование выбранной стратегии торговли и считающий статестические параметры (+ графики) данной стратегии. Реализация только для дельта-хеджирования.
***
2. **src/working_with_files/preprocessing.py**\ (source='moex_fixing')"
   > Парсинг MOEX fixing данных из xml файлов + скачивание этих файлов. Запасной бесплатный источник.
***
3. **src/working_with_files/preprocessing.py (source='bbg')**\ 
   > Парсинг дневных данных Блумберга. Не поддерживает получение данных в онлайн-режиме (ограничение Блумберга).
***
4. **src/working_with_files/preprocessing.py (source='moex') или (source='rbi')**\
   > Парсинг часовых данных Мосбиржи + часовых данных валют на оффшоре (в Европе). Это самые полезные данные из спотовых.
***
5. **(!) generate_synthetic_fx.py**\ 
   > Генерация "синтетического курса" валюты. Есть валюты, которые отдельно не торгуются на оффшорных рынках, только локально. Но так как любая валютная пара A/B=A/C * C/B, имея спотовые данные A/B локальные и C / B оффшорные (или наоборот), можно генерировать новый FX курс. В таком курсе очень много неэффективностей в плане волатильности.
***
6. **(!) src/working_with_files/preprocessing.py get_option_prices(source='forts')**\
   > Самое интересное - парсинг опционных данных из файлов с forts. Автоматом они не скачиваются, только закупаются от биржи.
***

Парсинг телеграм-каналов и сайтов

Данные хранятся на google-диске (https://drive.google.com/drive/u/0/folders/1YDZU_Ol3vj2jdYe3opMRn_eYi8KxgFyO) в формате .csv

https://t.me/bitkogan - bitkogan.csv

https://t.me/russianmacro - mmi.csv

https://t.me/warwisdom - War_Wealth_Wisdom.csv

https://t.me/themovchans - themovchans.csv

https://t.me/v_tsuprov - vts.csv

https://t.me/skybond - sky_bond.csv

https://t.me/alfawealth - Alfa_Wealth.csv

https://t.me/RSHB_Invest - rshb_invest.csv

https://t.me/headlines_quants - headlines_QUANTS.csv

https://t.me/cbrstocks - signal.csv

https://cbonds.ru/news/ - cbonds.csv

https://www.tinkoff.ru/invest/research/ - tinkoff.csv
