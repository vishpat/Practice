(ns project-euler.problem21
  (require [project-euler.util :as util]))

(defn d-n
  [n]
  (reduce + (rest (util/get-all-factors n)))
  )

(defn solve
  []
  (loop [n 0 sum 0]
    (if (= n 10000) sum 
      (let [a (d-n n) b (d-n a)]
        (if (and (not= a n) (= n b)) 
          (recur (inc n) (+ n sum))
          (recur (inc n) sum))
       )
      )
    )
  )
