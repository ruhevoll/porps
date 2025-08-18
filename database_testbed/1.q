/ read0 `C:/Users/ruhev/Documents/porps/database_testbed/data/apple.csv
AAPL:("DSISSS";enlist",") 0: `C:/Users/ruhev/Documents/porps/database_testbed/data/apple.csv
AAPL:update Close:"F"$1_'string Close, Open:"F"$1_'string Open,High:"F"$1_'string High, Low:"F"$1_'string Low from AAPL

show "Closing price statistics for Apple:"
show "Average:", (string avg t.Close)
show "Min: ", (string min t.Close)
show "Max: ", (string max t.Close)
