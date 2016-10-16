(ns project-euler.problem31)

(def coins '(1 2 5 10 20 50 100 200)
(def combinations (@atom {}))

(defn solve
  []
  (loop [amount 0]
    (if (= amount 201) (get @combinations 200)
      (do
       (map #(let [amount-combinations (get @combinations amount) 
                   coin-combination (get @combinations (- amount %))] 
                   (if (= % amount) (assoc @combinations amount (inc amount-combinations)) 
                       ((when (some? coin-combination) 
                        (assoc @combinations amount (+ amount-combinations coin-combinations)))))) 
            coins) 
       (recur (inc amount)) 
        )
   )
  )
