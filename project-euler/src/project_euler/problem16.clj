(ns project-euler.problem16 
  (:require [project-euler.util :as util]))

(def pow-2-map (atom {}))



(defn str-2-pow-n
  [n]
  (loop [product "2" index 0]
    (if (= index n) (util/calculate-str-num-digit-sum product)
      (do (recur (util/str-num-mult product "2") (inc index)))
    )
   )
  )


(defn str-2-pow-10

  )

(defn solve
  []
  (loop [product "2" index 0]
    (if (= index 10) (util/calculate-str-num-digit-sum product)
      (do (println product) (recur (util/str-num-mult product "2") (inc index)))
    )
   )
  )
