const vectors = {NORTH:[0,-1], SOUTH:[0,1], EAST:[1,0], WEST:[-1,0]};
const sameCell = (a,b) => a[0]===b[0] && a[1]===b[1];
const positions = g => [g.observation.farms[0].farmer,...g.observation.farms[0].hands];

export function operationFor(intent,game) {
  const from=positions(game)[intent.unit];
  if(!from)throw new Error('农工已离开，后续行动已停止');
  if(!intent.to)return intent.op || ['PASS'];
  if(sameCell(from,intent.to))return null;
  const [x,y]=from,[tx,ty]=intent.to;
  return [x!==tx?(x<tx?'EAST':'WEST'):(y<ty?'SOUTH':'NORTH')];
}

// Compatibility for still-running older local servers. Only show the four
// animations when the result can be established from the two observations.
export function receipt(before,after,unit,action) {
  if(after.animations) return after.animations.find(e=>e.unit===unit && e.op===action[0]);
  if(before.observation.day!==after.observation.day)return null;
  const start=positions(before)[unit],end=positions(after)[unit];
  if(!start || !end)return null;
  const [x,y]=start, old=before.observation.farms[0].tiles[y][x], now=after.observation.farms[0].tiles[y][x];
  const op=action[0],delta=vectors[op];
  let item='';
  if(delta && sameCell(end,[x+delta[0],y+delta[1]]))return {unit,op,start,end};
  if(op==='PLANT' && !old && now?.crop===action[1])item=now.crop;
  if(op==='WATER' && old?.kind==='PLANT' && !old.watered_today && now?.watered_today)item=now.crop;
  if(op==='HARVEST'){
    const bag=after.observation.private.inventories[unit];
    item=Object.keys(bag).find(k=>bag[k]>(before.observation.private.inventories[unit][k]||0));
  }
  return item?{unit,op,start,end,item}:null;
}

// A market fill changes cash even when the selected farm action failed.
// Check the result on the unit itself so a failed action still stops its route.
function applied(before,after,unit,op) {
  const name=op[0];
  if(name==='PASS')return true;
  if(name in vectors || ['PLANT','WATER','HARVEST'].includes(name))
    return Boolean(receipt(before,after,unit,op));
  const [x,y]=positions(before)[unit];
  const prev=before.observation.farms[0].tiles[y][x];
  const next=after.observation.farms[0].tiles[y][x];
  if(name==='DIG')return prev!=null && next==null;
  if(name==='BUILD_COOP' || name==='BUILD_PASTURE')
    return prev==null && next?.kind===name.slice(6);
  if(name==='CARE')return !prev?.cared_today && next?.cared_today;
  const bag=before.observation.private.inventories[unit];
  const result=after.observation.private.inventories[unit];
  return [...Object.keys(bag),...Object.keys(result)].some(key=>(bag[key]||0)!==(result[key]||0));
}

export class ActionQueue {
  constructor({read,step,onChange,onStop}) {
    this.read=read;this.step=step;this.onChange=onChange;this.onStop=onStop;
    this.items=[];this.running=false;
  }
  push(intent) {
    if(this.items.length>=24){this.onStop('队列已满，先等农夫完成几项行动');return;}
    this.items.push(intent);this.onChange();
    if(!this.running)void this.run();
  }
  cancel(message='后续行动已取消') {
    this.items.length=0;this.onChange();this.onStop(message);
  }
  async run() {
    this.running=true;
    try {
      while(this.items.length) {
        const intent=this.items[0],before=this.read(),op=operationFor(intent,before);
        if(!op){this.items.shift();continue;}
        const units=positions(before);
        const action={farmer:intent.unit===0?op:['PASS'],hands:units.slice(1).map((_,i)=>i+1===intent.unit?op:['PASS'])};
        await this.step(action);
        const after=this.read();
        if(before.id!==after.id || after.observation.step!==before.observation.step+1)
          throw new Error('回合状态已变化，请刷新后继续');
        if(after.done || before.observation.day!==after.observation.day){
          this.cancel(after.done?'赛季已结束':'新的一天开始，队列已停止');break;
        }
        if(this.items[0]!==intent)continue;
        if(!applied(before,after,intent.unit,op)){
          this.cancel('农夫行动未生效，后续行动已停止。交易结果见田间日志。');break;
        }
        if(!intent.to || sameCell(positions(after)[intent.unit],intent.to))this.items.shift();
        this.onChange();
      }
    } catch(e){this.cancel(e.message);}
    finally {this.running=false;this.onChange();}
  }
}
