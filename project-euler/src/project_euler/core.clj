(ns project-euler.core)

(defn digit-at
  [number index]
  (if (< index (count number))
      (Integer/parseInt (str (.charAt number (- (count number) (+ index 1))))) 
      0)
  )

(defn sub-1-str
  [m]
    (str (- (Integer/parseInt m) 1)) 
)

(defn add-str
    [m n]
  (loop [num1 m num2 n index 0 carry 0 result ""]
    (cond
      (and (>= index (count num1)) (>= index (count num2))) (str (if (not= carry 0) carry "") result)
      :else 
      (let [d1 (digit-at num1 index)  
          d2 (digit-at num2 index)
          total (+ d1 d2 carry) 
          result-digit (mod total 10)
          next-carry (quot total 10)] 
          (recur num1 num2 (+ index 1) next-carry (str (str result-digit) result))  
      )
    )
  )
)

(defn multiply-str
  [m n]
  (loop [num1 m num2 n result ""]
    (cond (= num1 "0") "0" (= num2 "0") "0" (= num1 "1") (add-str result num2) 
          :else (recur (sub-1-str num1) num2 (add-str result num2)))  
  )
)


(defn factorial
  [n]
  (
   loop [x n result "1"] 
   (if (= x "1") result ( recur (sub-1-str x) (multiply-str x result))
   )
  )
)

(defn problem-20
  []
  (reduce + (map #(Integer/parseInt %) (map str (seq (factorial "100")))))
  )
