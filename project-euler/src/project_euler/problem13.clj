(ns project-euler.problem13
  (:require [clojure.java.io :as io]))

(defn sum-column 
  [data-file column]
  (with-open [rdr (io/reader data-file)]
    (reduce (fn [total line] (reduce + total (map #(Integer/parseInt %) (list (str (nth (seq line) column)))))) 0 (line-seq rdr))
  )
 ) 
