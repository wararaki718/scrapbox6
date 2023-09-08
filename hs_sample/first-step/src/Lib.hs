module Lib
    ( someFunc
    ) where

someFunc :: IO ()
someFunc = do
    let result = (succ 9::Int) + (max 5 4) + 1;
    putStrLn $ show result
