(ns project-euler.util)

(defn str-num-digit-at
  [number index]
  (if (< index (count number))
      (Integer/parseInt (str (.charAt number (- (count number) (+ index 1))))) 
      0)
  )

(defn str-num-sub-1
  [m]
    (str (- (Integer/parseInt m) 1)) 
)

(defn str-num-add
    [m n]
  (loop [num1 m num2 n index 0 carry 0 result ""]
    (cond
      (and (>= index (count num1)) (>= index (count num2))) (str (if (not= carry 0) carry "") result)
      :else 
      (let [d1 (str-num-digit-at num1 index)  
          d2 (str-num-digit-at num2 index)
          total (+ d1 d2 carry) 
          result-digit (mod total 10)
          next-carry (quot total 10)] 
          (recur num1 num2 (+ index 1) next-carry (str (str result-digit) result))  
      )
    )
  )
)

(defn str-num-mult
  [m n]
  (loop [num1 m num2 n result ""]
    (cond (= num1 "0") "0" (= num2 "0") "0" (= num1 "1") (str-num-add result num2) 
          :else (recur (str-num-sub-1 num1) num2 (str-num-add result num2)))  
  )
)

(defn str-num-factorial
  [n]
  (
   loop [x n result "1"] 
   (if (= x "1") result ( recur (str-num-sub-1 x) (str-num-mult x result))
   )
  )
)

(defn str-2-pow-n
  [n]
  (loop [product "1" index 0]
    (if (= index n) product
      (recur (str-num-add product product) (inc index))
    )
   )
  )

(defn calculate-str-num-digit-sum
  [str-num]
  (reduce + (map #(Integer/parseInt %) (map str (seq str-num))))
  )


(defn get-prime-factors
  [n]
  (loop [num n factor 2 factors '()]
    (cond (> factor num) factors  
          (= (mod num factor) 0) (recur (quot num factor) 2 (cons factor factors)) 
          :else (recur num (+ factor 1) factors))
  )
  )

(defn get-all-factors
  [n]
  (loop [num n factor 1 factors '()]
    (cond (> factor (quot num 2)) (cons n factors)  
          (= 0 (mod num factor)) (recur num (+ factor 1) (cons factor factors)) 
          :else (recur num (+ factor 1) factors))
  )
  )
