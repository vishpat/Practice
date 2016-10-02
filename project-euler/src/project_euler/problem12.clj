(ns project-euler.problem12
  (require [project-euler.util :as util]))

(defn triangle-number
  [n]
  (reduce + n (range n))
  )

(defn solve
  []
  (loop [num 1]
   (let [triangle-num (triangle-number num) divisors (util/get-all-factors triangle-num) divisor-count (count divisors)]
     (if (= divisor-count 50) num
       (recur (+ num 1)))
   )
  )
  )
