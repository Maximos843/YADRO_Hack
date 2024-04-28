# Решение хакатона от компании YADRO по прогнозированию погодных условий


Были предоставлены данные в виде карты 30х30 за 43 часа с отметкой в 1 час. Показатели: температура, высота, давление, облачность, влажность, скорость и направление ветра. Было необходимо спрогнозировать эти параметры на следующие 5 часов. В качестве целевой метрики использовалась MAPE.


Для компиляции докер файла:
```make build```


Для запуска контейнера:
```make run```