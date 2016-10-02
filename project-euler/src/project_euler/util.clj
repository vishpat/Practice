(ns project-euler.util)

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
