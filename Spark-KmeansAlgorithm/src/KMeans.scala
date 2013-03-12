import java.util.Random
import spark.SparkContext._
import spark.util.Vector
import scala.collection.mutable.HashMap
import scala.collection.mutable.HashSet

/*
 * K-means algorithm in Machine Learning (Clustering algorithm)
 * Running on Spark framework, which is Hadoop/MapReduce-like
 *  with caching data in the memory across cluster 
 * Can be deployed on AWS
 * The steps include:
 * 	1) randomly generate a number of D-dimensional points 
 *  2) randomly generate K centers 
 *  3) iteratively move centers until hitting convergence
 */
object KMeans {
	val N = 1000
	val R = 1000  	
	val D = 10
	val K = 10
	val threshold = 0.0005
	val rand = new Random(42)
  	
	/* generate data points */
	def generatePoints = {
	    def generatePoint(i: Int) = {
	      Vector(D, _ => rand.nextDouble * R)
	    }
	    Array.tabulate(N)(generatePoint)
	}

	/* find out the closest point around centers */
	def closestPoint(p: Vector, centers: HashMap[Int, Vector]): Int = {
		var index = 0
		var closest_index = 0
		var closest = Double.PositiveInfinity

		for (i <- 1 to centers.size) {
			val cur_val = centers.get(i).get
			val temp_distance = p.squaredDist(cur_val)
			if (temp_distance < closest) {
				closest = temp_distance
				closest_index = i
			}
		}

		return closest_index
	}

	/* main process */
	def main(args: Array[String]) {
	    val data = generatePoints
		var points = new HashSet[Vector]
		var centers = new HashMap[Int, Vector]
		var temp_distance = 1.0

		while (points.size < K) {
			points.add(data(rand.nextInt(N)))
		}
		val ite = points.iterator
		for (i <- 1 to points.size) {
			centers.put(i, ite.next())
		}
		println("Old centers: " + centers)

		/* iterate until hitting convergence */
		while(temp_distance > threshold) {
			var closest = data.map (p => (closestPoint(p, centers), (p, 1)))
			var mappings = closest.groupBy[Int] (x => x._1)
			var point_stat = mappings.map(pair => pair._2.reduceLeft [(Int, (Vector, Int))] {case ((id1, (x1, y1)), (id2, (x2, y2))) => (id1, (x1 + x2, y1+y2))})
			var new_points = point_stat.map {mapping => (mapping._1, mapping._2._1/mapping._2._2)}
			temp_distance = 0.0
			for (mapping <- new_points) {
				temp_distance += centers.get(mapping._1).get.squaredDist(mapping._2)
			}
			for (newP <- new_points) {
				centers.put(newP._1, newP._2)
			}
		}
		println("New centers: " + centers)
	}
}