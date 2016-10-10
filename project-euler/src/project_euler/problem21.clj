(ns project-euler.problem21
  (require [project-euler.util :as util]))

(defn factor-sum
  [num]
  (reduce + (rest (util/get-all-factors num)))
  )

(defn solve
  []
  (loop [num 0 sum 0]
    (println num " " sum)
    (if (= num 10000) sum 
      (let [num-sum (factor-sum num) sum-num-sum (factor-sum num-sum)]
        (if (= num-sum sum-num-sum) 
          (recur (inc num) (+ num-sum sum))
          (recur (inc num) sum))
       )
      )
    )
  )
