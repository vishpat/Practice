(ns project-euler.problem12
  (require [project-euler.util :as util]))

(defn solve
  []
  (loop [triangle-num 1 index 1]
   (let [divisors (util/get-all-factors triangle-num) divisor-count (count divisors)]
     (println "index " index " triangle-num " triangle-num)
     (if (>= divisor-count 5) num
       (recur (+ triangle-num (+ index 1)) (+ index 1)))
   )
  )
  )
