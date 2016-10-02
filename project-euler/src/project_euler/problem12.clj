(ns project-euler.problem12
  (require [project-euler.util :as util]))

(defn solve
  []
  (loop [triangle-num 1 index 1]
   (let [divisors (util/get-all-factors triangle-num) divisor-count (count divisors)]
     (if (>= divisor-count 500) triangle-num 
       (recur (+ triangle-num (+ index 1)) (+ index 1)))
   )
  )
  )
