(ns project-euler.problem18
 (:require [clojure.java.io :as io]) )

(def data (atom {}))
(def line-cnt (atom 0))

(defn read-data-file
  [data-file]
  (with-open [rdr (io/reader data-file)]
    (doseq [line (line-seq rdr)]
            (do 
                (let [line-numbers (map #(Integer/parseInt %) (clojure.string/split line #"\s+"))
                      x-idx (atom 0)]
                   (doseq [n line-numbers]
                     (do
                        (swap! data assoc (list @x-idx @line-cnt) n)
                        (swap! x-idx inc) 
                      )
                   ) 
                  )
                (swap! line-cnt inc)
                )
            )
  )
)

(defn solve
  []
  (do
    (read-data-file "problem18.txt")
    (loop [x 0 y 0 max-total (get @data (list 0 0))]
      (do
      (let [left-child-total  (if (< y (- @line-cnt 1)) (+ max-total (get @data (list x (inc y)))) 0) 
            right-child-total (if (< y (- @line-cnt 1)) (+ max-total (get @data (list (inc x) (inc y)))) 0)] 
            (cond (= y  (- @line-cnt 1)) max-total
                  (> left-child-total right-child-total) (recur x (inc y) left-child-total)
                  :else (recur (inc x) (inc y) right-child-total)
          )
        )
      )
    )
  )
)
