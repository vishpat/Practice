(ns project-euler.problem16 
  (:require [project-euler.util :as util]))

(defn solve
  []
  (util/calculate-str-num-digit-sum (util/str-2-pow-n 1000))
 ) 
