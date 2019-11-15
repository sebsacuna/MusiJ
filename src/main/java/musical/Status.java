package musical;

public class Status {
    public static int total = 0;
    public static int progress = 1;
    public static boolean done = false;

    public Status(int initial, int total){
        this.total = total;
        this.progress = initial;
        this.done = false;
    }
}
