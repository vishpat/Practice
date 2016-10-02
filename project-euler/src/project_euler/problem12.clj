(ns project-euler.problem12
  (require [project-euler.util :as util]))

(defn divisor-cnt
  [n]
  (let [max-idx (Math/sqrt n)]
     (loop [cnt 0 idx 1]
      (cond (= n 1) 1
            (> idx (+ max-idx)) cnt
            (= 0 (mod n idx)) (recur (+ cnt 2) (+ idx 1))
            :else (recur cnt (+ idx 1)))
       )
   )
  )
  

(defn solve
  []
  (loop [triangle-num 1 index 1]
   (let [cnt (divisor-cnt triangle-num)]
     (if (>= cnt 500) triangle-num 
       (recur (+ triangle-num (+ index 1)) (+ index 1)))
   )
  )
  )
