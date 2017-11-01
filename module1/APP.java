package m2;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.storage.StorageLevel;
import org.spark_project.guava.collect.Lists;

import scala.Tuple2;

public class APP {
	public static void main(String[] args) {
		double start = System.currentTimeMillis();
		double end = 0.0;
		
		String filePath = args[0];
		String friend_file = args[1];
		String distance_file = args[2];
		double threshold = Double.parseDouble(args[3]);
		int k = Integer.parseInt(args[4]);

		SparkConf conf = new SparkConf().setAppName("friend recommand");
		JavaSparkContext ctx = new JavaSparkContext(conf);
		//friend relation data.
		JavaRDD<String> friends_data = ctx.textFile(filePath+friend_file);
		//friend distance data.
		JavaRDD<String> distance_data = ctx.textFile(filePath+distance_file);
		//RDD<id, friend> pair data.
		JavaPairRDD<Integer, Integer> friend_pair = friends_data.flatMapToPair(p -> {
			int id = Integer.parseInt(p.split("\t")[0].trim());
			int friend = Integer.parseInt(p.split("\t")[1].trim());
			List<Tuple2<Integer, Integer>> result = new ArrayList();
			result.add(new Tuple2(id, friend));
			result.add(new Tuple2(friend, id));
			return result.iterator();
		}).persist(StorageLevel.MEMORY_AND_DISK_SER_2());
		//free friend relation data.
		friends_data.unpersist();
		//RDD<id, friend's list> pair data.
		JavaPairRDD<Integer, ArrayList<Integer>> friend_list = friend_pair.groupByKey().flatMapToPair(g -> {
			List<Tuple2<Integer, ArrayList<Integer>>> result = new ArrayList<>();
			result.add(new Tuple2<>(g._1(), Lists.newArrayList(g._2())));
			return result.iterator();
		}).persist(StorageLevel.MEMORY_AND_DISK_SER_2());
		//free friend pair data.
		friend_pair.unpersist();
		
		//Broadcast list<id,friend's list>
		Broadcast<List<Tuple2<Integer, ArrayList<Integer>>>> friendlist_broad = ctx.broadcast(friend_list.sortByKey().collect());
		
		//RDD<id,distance pair> data
		JavaPairRDD<Integer, Tuple2<Double, Double>> id_location = distance_data.flatMapToPair(p -> {
			int id = Integer.parseInt(p.split("\t")[0].trim());
			double x = Double.parseDouble(p.split("\t")[1].trim());
			double y = Double.parseDouble(p.split("\t")[2].trim());
			List<Tuple2<Integer, Tuple2<Double, Double>>> result = new ArrayList();
			result.add(new Tuple2(id, new Tuple2(x, y)));
			return result.iterator();
		}).persist(StorageLevel.MEMORY_AND_DISK_SER_2());
		//free distance data
		distance_data.unpersist();
		//Broadcast list<id,distance pair> data
		Broadcast<List<Tuple2<Integer, Tuple2<Double, Double>>>> distancelist_broad = ctx.broadcast(id_location.sortByKey().collect());
		//free RDD<id,distance pair> data
		id_location.unpersist();
		//RDD<unfriend_pair, intersection friend number> data
		JavaPairRDD<Tuple2<Integer, Integer>, Integer> friend_intersection = friend_list.flatMapToPair(p -> {
			final int not_friend = -Integer.MIN_VALUE;
			List<Tuple2<Tuple2<Integer, Integer>, Integer>> result = new ArrayList<>();
			List<Integer> friendlist = p._2();
			int friend_list_size = friendlist.size();
			for (int i = 0; i < friend_list_size - 1; i++) {
				for (int j = i + 1; j < friend_list_size; j++) {
					if (friendlist.get(i) < friendlist.get(j))
						result.add(new Tuple2<>(new Tuple2<>(friendlist.get(i), friendlist.get(j)), 1));
					else
						result.add(new Tuple2<>(new Tuple2<>(friendlist.get(j), friendlist.get(i)), 1));
				}
			}
			for(int i: p._2)
				result.add(new Tuple2(new Tuple2(p._1, i), not_friend));
			return result.iterator();
		}).reduceByKey((a, b) -> a + b).filter(p -> p._2() > 0).persist(StorageLevel.MEMORY_AND_DISK_SER_2());
		//free friend list data
		friend_list.unpersist();
		
		//RDD<jaccard similar, unfriend pair> data
		JavaPairRDD<Double,Tuple2<Integer, Integer>> friend_jsimilar = friend_intersection.flatMapToPair(p -> {
			List<Tuple2<Double,Tuple2<Integer,Integer>>> result = new ArrayList<>();
			double jaccard=(double)p._2/(friendlist_broad.getValue().get(p._1._1)._2.size()+friendlist_broad.getValue().get(p._1._2)._2.size()-p._2);
			if(jaccard>threshold) {
				result.add(new Tuple2<Double,Tuple2<Integer,Integer>>(jaccard, p._1));
			}
			return result.iterator();
		}).persist(StorageLevel.MEMORY_AND_DISK_SER_2());
		
		//free intersection RDD data.
		friend_intersection.unpersist();
		//free Broad list.
		friendlist_broad.unpersist();
		
		//<distance, <jaccard, unfriend pair>> list
		List<Tuple2<Double, Tuple2<Double, Tuple2<Integer, Integer>>>> distance_jsimilar = friend_jsimilar.flatMapToPair(p -> {
			List<Tuple2<Double, Tuple2<Double, Tuple2<Integer, Integer>>>> result = new ArrayList();
			Tuple2<Integer, Integer> pair = p._2();
			Tuple2<Double, Double> location1 = distancelist_broad.value().get(pair._1())._2();
			Tuple2<Double, Double> location2 = distancelist_broad.value().get(pair._2())._2();
			double distance = Math.sqrt(Math.pow(location1._1 - location2._1, 2) + Math.pow(location1._2 - location2._2, 2));
			result.add(new Tuple2(distance, new Tuple2(p._1(), pair)));
			return result.iterator();
		}).sortByKey().take(k);
		
		//free jaccard RDD data
		friend_jsimilar.unpersist();
		//free Broadcast list.
		distancelist_broad.unpersist();
		
//		for(Tuple2<Double, Tuple2<Double, Tuple2<Integer, Integer>>> i : distance_jsimilar) {
//			System.out.println(i._2._2._1 + "\t" + i._2._2._2 + "\t" + i._1 + "\t" + i._2._1 + "\n");
//		}
		try {
			BufferedWriter bw = new BufferedWriter(new FileWriter("./"+ "result " + threshold + "-" + k + ".txt", true));
			for (Tuple2<Double, Tuple2<Double, Tuple2<Integer, Integer>>> i : distance_jsimilar)
				bw.write(i._2._2._1 + "\t" + i._2._2._2 + "\t" + i._1 + "\t" + i._2._1 + "\n");
			bw.close();
		} catch (IOException e) {
			e.printStackTrace();
		}

		ctx.close();
		end = System.currentTimeMillis();
		System.out.println(end - start);
	}
}