(ns project-euler.problem18
 (:require [clojure.java.io :as io]) )

(def data (atom {}))
(def total-tree (atom {}))
(def line-cnt (atom 0))

(defn get-data-val
  [x y]
  (get @data (x y))
  )

(defn get-left-child-data
  [x y]
  (get @data (x (inc y)))
)

(defn get-left-child-data
  [x y]
  (get @data ((inc x) (inc y)))
)

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

(defn bfs
  []
  (let [stack (atom (vec (list 0 0 (get-data-val 0 0))))]
    (loop [stack (vec (list 0 0 (get-data-val 0 0))) max-total 0]
      (if (= 0 (count stack)) max-total
            (let [tos (nth stack 0)
                  tos-x (nth tos 0)
                  tos-y (nth tos 1)
                  tos-total (nth tos 2)
                  next-x (inc x)
                  next-y (inc y)
                  left-child-val (get-left-child-data tos-x next-y)
                  left-child-total (+ tos-total left-child-val)
                  right-child-val (get-left-child-data next-x next-y)
                  right-child-total (+ tos-total right-child-val)
                  left-child (list tos-x next-y left-child-total) 
                  right-child (list next-x next-y right-child-total)
                  next-max-total (max max-total left-child-total right-child-total)]
              (do
              (when (some? (get-data-val tos-x next-y)) (conj stack left-child))
              (when (some? (get-data-val next-x next-y)) (conj stack right-child))
              (recur next-max-total))        
              )
      )

      )
   )
)

(defn solve
  []
  (do
    (read-data-file "problem18.txt")
    (bfs)  
  )
)
