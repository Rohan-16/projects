import java.awt.Color;
import java.util.Random;

public class Life{
    private int size;
    private int iteration;
    private String t;
    private Picture pic;
    private int [][] cells; 
    public Life(int s,String t){
        this.size=s;
        this.t=t;
        this.cells= new int[s][s];
        
        pic=new Picture(s*10,s*10);
        if (this.t.equals("g")){
            cells[26][2]=1;
            cells[24][3]=1;
            cells[26][3]=1;
            cells[14][4]=1;
            cells[15][4]=1;
            cells[22][4]=1;
            cells[23][4]=1;
            cells[36][4]=1;
            cells[37][4]=1;
            cells[13][5]=1;
            cells[17][5]=1;
            cells[22][5]=1;
            cells[23][5]=1;
            cells[36][5]=1;
            cells[37][5]=1;
            cells[2][6]=1;
            cells[3][6]=1;
            cells[12][6]=1;
            cells[18][6]=1;
            cells[22][6]=1;
            cells[23][6]=1;
            cells[2][7]=1;
            cells[3][7]=1;
            cells[12][7]=1;
            cells[16][7]=1;
            cells[18][7]=1;
            cells[19][7]=1;
            cells[24][7]=1;
            cells[26][7]=1;
            cells[18][8]=1;
            cells[12][8]=1;
            cells[26][8]=1;
            cells[13][9]=1;
            cells[17][9]=1;
            cells[14][10]=1;
            cells[15][10]=1;
            

            }
            else  {
                for(int b=0;b<this.size;b++){
                    for(int c=0;c<this.size;c++){
                        cells[b][c]=new Random().nextInt(2); 
                    }

                }

            }

    }
    public void generate(){
           int[][] next= new int[this.size][this.size];
       
            for(int x=0;x<this.size;x++){
                for(int y=0;y<this.size;y++){
                    int neighbour=0;
                    for(int k=-1;k<=1;k++){
                        for(int l=-1;l<=1;l++){
                            neighbour += cells[(x+k+this.size)%this.size][(y+l+this.size)%this.size];
                            //neighbour+=cells[x+k][y+l];
                        }
                    }
                    neighbour-=cells[x][y];
                    if ((cells[x][y]==1) && ((neighbour==2)||(neighbour ==3))) next[x][y]=1;
                    else if((cells[x][y]==1) && ((neighbour > 3))) next[x][y]=0;
                    else if((cells[x][y]==1) && ((neighbour < 2))) next[x][y]=0;
                    else if ((cells[x][y] ==0) && (neighbour ==3)) next[x][y] =1;
                    else next[x][y] =cells[x][y];
                }
            }
            this.cells=next;
       
    }        

     public void draw() {
         
        for (int i = 0; i < this.size; i++) {
            for (int j = 0; j < this.size; j++) {
                if ((cells[i][j])==0){ 
                    for (int q=i*10;q<((i*10)+10);q++){
                        for(int v=j*10;v<((j*10)+10);v++){
                        pic.set(q, v, Color.WHITE);
                    }
                }
                }
                else{
                     for (int q=i*10;q<((i*10)+10);q++){
                        for(int v=j*10;v<((j*10)+10);v++){
                        pic.set(q, v, Color.BLACK);
                    }
                }
                }
            }
        }
        
        pic.show();
        
    
    }
    public static void main(String[] args){
        int s=Integer.parseInt(args[0]);
        int i=Integer.parseInt(args[1]);
        String t=(args[2]);
        Life life=new Life(s,t);
        for (int z = 0; z < i; z++) {
            try { Thread.sleep(100); } // so we can see what's happening!
            catch (Exception ex) { /* ignore */ }
            
            life.generate();
            life.draw();
            
        }
    
    
    
    
    }
    

    
    




}