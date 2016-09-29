(ns project-euler.problem13
  (:require [clojure.java.io :as io]))

(defn sum-column 
  [data-file column]
  (with-open [rdr (io/reader data-file)]
    (reduce (fn [total line] (reduce + total (map #(Integer/parseInt %) (list (str (nth (seq line) column)))))) 0 (line-seq rdr))
  )
 )

(defn solve 
  [data-file]
  (loop [carry 0 column 49 digits ""]
    (if (< column 0) (str (str carry) digits) 
      (let [column-total (+ (sum-column data-file column) (mod carry 10)) next-carry (+ (quot carry 10) (quot column-total 10))]
          (recur next-carry (- column 1) (str (str (mod column-total 10)) digits)) 
      ) 
    )
  )
)
