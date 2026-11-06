import { opponents } from './opponents.js';
import { ActionQueue, receipt } from './continuous.js';
import { itemSprite, sprite } from './pixel-scene.js';
import { renderLayout } from './interface.js';
const $ = (s, el = document) => el.querySelector(s);
const app = $('#app'), dialog = $('#dialog');
let game, health, busy = false, resting = false, farmView = 0, unit = 0, selected = [4, 4], tab = 'seed', section = 'farm';
let orders = [], actions = {}, crop = 'CARROT', logOpen = false;
let gridVisible=false, zoomed=false, previousPositions=[], marketQuantity=1;
let continuous=true, motion=null;
const queueDriver=new ActionQueue({read:()=>game,step:advance,onChange:()=>render(),onStop:toast});
function queueLabel(){
 const job=queueDriver.items[0];
 if(!job)return '等待行动';
 const name=job.to?'前往 ('+job.to.join(', ')+')':opText(job.op);
 return name+' · '+queueDriver.items.length+' 项';
}
const names = {WHEAT:'小麦',CARROT:'胡萝卜',TOMATO:'番茄',STRAWBERRY:'草莓',MELON:'甜瓜',EGG:'鹅蛋',MILK:'牛奶',WOOL:'羊毛',FERTILIZER:'肥料',GOOSE:'鹅',COW:'奶牛',SHEEP:'绵羊'};
const crops = ['CARROT','WHEAT','TOMATO','STRAWBERRY','MELON'];
const seedPrices = {WHEAT:10,CARROT:20,TOMATO:50,STRAWBERRY:100,MELON:80};
const ages = {WHEAT:2,CARROT:2,TOMATO:8,STRAWBERRY:10,MELON:10};
const opNames = {PASS:'休息',NORTH:'向北移动',SOUTH:'向南移动',WEST:'向西移动',EAST:'向东移动',PLANT:'种植',WATER:'浇水',HARVEST:'收获',DIG:'清理',DROP:'全部入仓',PICKUP:'取出',PLACE:'放置',FERTILIZE:'施肥',BUILD_COOP:'建鹅舍',BUILD_PASTURE:'建牧场',FEED:'喂养',CARE:'照料',COLLECT_FERTILIZER:'收集肥料',BUY_SEED:'买种子',BUY_PRODUCT:'买物资',BUY_ANIMAL:'买牲畜',SELL:'出售',HIRE:'雇佣农工',BUY_LAND:'扩建土地'};
const money = n => Number(n).toLocaleString('en-US', {maximumFractionDigits:0});
const esc = value => String(value).replace(/[&<>"']/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const iconPaths = {
 sprout:'<path d="M12 21v-9M12 15C4 16 2 9 3 5c7 0 10 4 9 10Zm0-4c0-6 4-9 9-8 0 7-4 10-9 8Z"/>',
 grid:'<rect x="3" y="3" width="7" height="7" rx="2"/><rect x="14" y="3" width="7" height="7" rx="2"/><rect x="3" y="14" width="7" height="7" rx="2"/><rect x="14" y="14" width="7" height="7" rx="2"/>',
 market:'<path d="m3 9 2-6h14l2 6M4 9v12h16V9M3 9c0 4 5 4 5 0 0 4 8 4 8 0 0 4 5 4 5 0M9 21v-7h6v7"/>',
 book:'<path d="M12 5v16M12 5C8 2 4 3 2 4v15c4-1 7-1 10 2 3-3 6-3 10-2V4c-3-1-7-2-10 1Z"/>',
 cpu:'<rect x="5" y="5" width="14" height="14" rx="4"/><path d="M9 1v4m6-4v4M9 19v4m6-4v4M1 9h4m-4 6h4m14-6h4m-4 6h4"/><rect x="9" y="9" width="6" height="6" rx="1"/>',
 sun:'<circle cx="12" cy="12" r="4"/><path d="M12 1v2m0 18v2M1 12h2m18 0h2M4 4l2 2m12 12 2 2M4 20l2-2M18 6l2-2"/>',
 arrow:'<path d="M4 12h16m-6-6 6 6-6 6"/>',
 water:'<path d="M12 2S5 10 5 15a7 7 0 0 0 14 0c0-5-7-13-7-13Z"/><path d="M8 15a4 4 0 0 0 4 4"/>',
 cut:'<path d="M20 3c1 8-4 13-12 11L4 21l-2-2L16 5l4-2Z"/>',
 shovel:'<path d="m14 9-8 8M12 4l8 8m-8-8 3-3 8 8-3 3M6 14l4 4-6 4H1l1-4 4-4Z"/>',
 box:'<path d="m3 7 9-4 9 4v12l-9 4-9-4V7Zm0 0 9 4 9-4M12 11v12M7 5l10 4"/>',
 coin:'<circle cx="12" cy="12" r="9"/><path d="M15 8h-4a2 2 0 0 0 0 4h2a2 2 0 0 1 0 4H9m3-10v12"/>',
 plus:'<path d="M12 5v14M5 12h14"/>',
 chevron:'<path d="m9 5 7 7-7 7"/>',
 clock:'<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
 reset:'<path d="M3 10a9 9 0 1 1 1 7M3 3v7h7"/>',
 lock:'<rect x="5" y="10" width="14" height="11" rx="3"/><path d="M8 10V7a4 4 0 0 1 8 0v3m-4 5v2"/>',
 person:'<circle cx="12" cy="7" r="4"/><path d="M4 22v-3a8 8 0 0 1 16 0v3"/>',
 check:'<path d="m5 12 4 4L19 6"/>',
 info:'<circle cx="12" cy="12" r="9"/><path d="M12 11v6m0-10v.1"/>',
 download:'<path d="M12 2v13m-5-5 5 5 5-5M4 17v5h16v-5"/>',
 close:'<path d="m6 6 12 12M6 18 18 6"/>',
 history:'<path d="M3 10a9 9 0 1 1 1 7M3 3v7h7m2-3v5l4 2"/>'
};
const icon = (name, size=20) => `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${iconPaths[name] || iconPaths.sprout}</svg>`;
const produce = itemSprite;
const opText = op => `${opNames[op[0]] || op[0]}${op[1] ? ' · '+(names[op[1]] || op[1]) : ''}${op[2] ? ' ×'+op[2] : ''}`;
const obs = () => game.observation;
const own = () => obs().farms[0];
const pos = () => unit === 0 ? own().farmer : own().hands[unit-1] || own().farmer;
const bag = () => obs().private.inventories[unit] || {};
const shedAccess = () => [4,5].includes(pos()[0]) && [4,5].includes(pos()[1]);
async function api(path, body) {
 const r = await fetch('/api/'+path, body === undefined ? {} : {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
 const data = await r.json();
 if (!r.ok) throw new Error(data.error || '暂时无法连接游戏服务');
 return data;
}
let toastTimer;
function toast(message) { $('#toast').textContent = message; $('#toast').classList.add('show'); clearTimeout(toastTimer); toastTimer = setTimeout(()=>$('#toast').classList.remove('show'), 4200); }
function remember() { try { localStorage.setItem('farm-rl-session', game.id); } catch {} }
function btn(action, label, cls='', attrs='') { return `<button class="${cls}" data-do="${action}" ${attrs}>${label}</button>`; }
function openDialog(title, body, wide=false) {
 dialog.innerHTML = `<div class="dialog-heading"><div><span class="eyebrow">FIELDNOTE / FIELD GUIDE</span><h2>${title}</h2></div>${btn('close',icon('close'),'icon-button','aria-label="关闭"')}</div><div class="dialog-body ${wide?'wide':''}">${body}</div>`;
 if (!dialog.open) dialog.showModal();
}
function tileDetails() {
 const t=obs().farms[farmView].tiles[selected[1]][selected[0]];
 if(t==='LOCKED') return {title:'尚未开垦的土地',text:'可以通行；扩建后才能种植和建造。',symbol:'lock'};
 if(!t) return {title:'一块等待播种的土地',text:'空闲地块 · 农夫到达这里后即可播种。',symbol:'sprout'};
 if(t.kind==='PLANT') return {title:names[t.crop],text:`生长 ${obs().day-t.planted_day} 天 · ${t.watered_today?'今日已浇水':'今日待浇水'} · 产量 ${t.yield_units}`,symbol:'sprout'};
 if(t.kind==='WEED') return {title:'杂草',text:'移动到这里，使用清理腾出地块。',symbol:'shovel'};
 return {title:t.animal?names[t.animal]:t.kind==='COOP'?'空鹅舍':'空牧场',text:t.animal?`产量 ${t.yield_units} · ${t.fed_today?'今日已喂食':'今日待喂食'}`:'从仓库取出对应动物后在此放置。',symbol:'grid'};
}
function render() {
 if(!game)return;
 if(unit>own().hands.length)unit=0;
 const o=obs(), f=own(), other=o.farms[1],detail=tileDetails(),active=actions[unit]||['PASS'];
 const plants=f.tiles.flat().filter(t=>t?.kind==='PLANT'),pendingWater=plants.filter(t=>!t.watered_today).length;
 const oldScene=$('.scene-art');const sceneScroll=oldScene&&oldScene.scrollWidth>oldScene.clientWidth?{left:oldScene.scrollLeft,top:oldScene.scrollTop}:null;
 app.innerHTML=renderLayout({game,health,o,f,other,detail,active,plants,pendingWater,actions,orders,unit,selected,farmView,section,tab,logOpen,busy,continuous,queued:queueDriver.items.length,queueLabel:queueLabel(),motion,gridVisible,zoomed,previousPositions,marketQuantity,icon,btn,money,esc,opText,inventoryText,marketCards});
 if(zoomed){const scene=$('.scene-art');scene.scrollLeft=sceneScroll?.left??200;scene.scrollTop=sceneScroll?.top??0;}
 previousPositions=[];
}
function inventoryText(data) { return Object.entries(data).filter(([,v])=>v>0).map(([k,v])=>`${names[k]||esc(k)} <b>${v}</b>`).join(' · ') || '<span class="empty">空空的，期待下一次收获</span>'; }
function marketCards() {
 const list=tab==='seed'?crops:tab==='sell'?Object.keys(obs().market.prices):['FERTILIZER','WHEAT','GOOSE','COW','SHEEP'];
 return list.map(k=>{
  const carried = obs().private.inventories.reduce((n,bag)=>n+(bag[k]||0),0);
  const price=tab==='seed'?seedPrices[k]:({GOOSE:300,COW:400,SHEEP:500}[k]||obs().market.prices[k]);
  const note=tab==='seed'?`${ages[k]} 天起收获`:tab==='sell'?`仓库 ${obs().private.shed[k]||0} · 背包 ${carried}`:k==='WHEAT'?'喂养牲畜':k==='FERTILIZER'?'帮助作物增产':k==='GOOSE'?'需建造鹅舍':'需建造牧场';
  return `<article class="product-card"><div class="product-illustration ${k.toLowerCase()}">${produce(k,53)}</div><strong>${names[k]}</strong><small>${note}</small><div><span>$${money(price)}<small> / ${tab==='seed'?'颗':'份'}</small></span>${btn('trade-'+k,icon('plus',15),'add-button',`aria-label="${tab==='sell'?'出售':'购买'}${names[k]}" ${game.done||(tab==='sell'&&!obs().private.shed[k]&&!carried)?'disabled':''}`)}</div></article>`;
 }).join('');
}
function queue(op) {
 if(game.done)return;
 if(orders.length>=10)return toast('一个回合最多 10 笔订单，请先结算。');
 orders.push(op);
 render();
 toast(continuous?'订单已暂存，将随下一步行动结算。':'订单已暂存，推进回合时结算。');
}
function plan(op) { if(game.done) return; if(continuous){queueDriver.push({unit,op});return;} actions[unit]=op; render(); toast(`已计划：${opText(op)} · 空格 / 推进回合执行`); }
async function advance(actionOverride) {
 if(busy||game.done) return;
 const before=game;
 const units=actionOverride||{farmer:actions[0]||['PASS'],hands:own().hands.map((_,i)=>actions[i+1]||['PASS'])};
 // Take this turn's orders before awaiting the server. Orders added while the
 // request or animation is in progress belong to the following turn.
 const submitted={...units,market:orders.splice(0)};
 busy=true;render();
 try {
  const after=await api('step',{id:before.id,step:before.observation.step,action:submitted});
  game=after;actions={};
  if(unit>own().hands.length)unit=0;
  const executed=[submitted.farmer,...submitted.hands];
  const event=executed.map((op,i)=>receipt(before,after,i,op)).find(Boolean);
  let duration=120;
  if(event && farmView===0){
   const clip={PLANT:2,WATER:3,HARVEST:4}[event.op]||1;
   duration=[0,420,690,910,760][clip];
   motion={...event,clip,face:({EAST:1,NORTH:2,WEST:3,SOUTH:0}[event.op]??1),duration,time:performance.now()};
  }
  selected=queueDriver.items[0]?.to||[...pos()];
  remember();render();
  await new Promise(resolve=>setTimeout(resolve,duration));
  if(game.done)showResult();
 } catch(e) {
  const message=submitted.market.length ? e.message+' 请刷新并核对本回合交易记录。' : e.message;
  toast(message);
  if(continuous && queueDriver.running)throw new Error(message);
 }
 finally {motion=null;busy=false;render();}
}
function showRules() {
 openDialog('让每个回合，都长出一点收获。',`<p class="dialog-lead">两座农场，一个共享市场，30 天的经营比赛。你和 AI 同时提交行动，由官方引擎结算。</p><div class="guide-steps">${[['01','先买种子','在集市暂存种子订单，点另一块地，走第一步时一起购买。原地购买可点「只交易」。'],['02','在脚下播种','用方向键移动农夫；种植只作用于农夫当前地块。'],['03','记得浇水','种下当天必须浇水；之后连续两天不浇水会变成杂草。'],['04','把收获变成现金','收获进入背包。在中央仓库旁入仓，再出售；每天结束也会自动入仓。']].map(([n,t,p])=>`<div><span>${n}</span><section><h3>${t}</h3><p>${p}</p></section></div>`).join('')}</div><div class="rule-callout">${icon('coin',24)}<div><strong>胜负，只看最后的现金。</strong><p>结束时种子、作物、动物、背包和仓库都不折算成现金。记得提前出售。</p></div></div><details open><summary>时间与行动</summary><p>每天 24 回合。每回合每位农夫执行一个行动，同时可提交最多 10 笔市场订单。农夫行动先结算，市场订单后结算，所以当回合购买的种子要到下一回合才能种。默认共 719 次行动，最后可行动时间为第 30 天 22:00。</p></details><details><summary>土地、仓库与农工</summary><p>初始现金 $3,000，西北 5×5 地块免费解锁。扩地依次为东北 $1,000、西南 $2,000、东南 $4,000。锁定土地可以通行。中央四格 (4,4)、(5,4)、(4,5)、(5,5) 均能操作仓库；容量 100，超出部分在全部入仓时丢失。农工当天有效，价格依次为 1、1、2、3、5…，次日需重新雇佣。</p></details><details><summary>作物与动物</summary><p>小麦、胡萝卜 2 天起收获；番茄 8 天、草莓和甜瓜 10 天起收获。一次性作物收获后消失；番茄和草莓多次产出。动物需要先建鹅舍或牧场，再从仓库取出并放置，每天用背包里的小麦喂养。连续两天未喂会逃跑。</p></details><details><summary>市场与城镇</summary><p>双方使用同一个动态市场，销售增加供给、降低价格；城镇商店消耗库存。只有仓库里的商品可以出售。种子和动物固定售价，产品价格逐单位结算。城镇按规则逐步解锁商店，其需求由官方引擎处理。</p></details><p class="subtle">规则对应本项目固定的 kaggle-environments 1.32.7。无效或资金不足的合法格式行动，官方引擎会忽略。</p>`,true);
}
function showModel() { showNew(); }
function showNew() {
 const choice = game?.opponent || 'starter';
 const cards = opponents.map(ai => `<label class="ai-choice"><input type="radio" name="opponent" value="${ai.id}" ${ai.id===choice?'checked':''}>${sprite(ai.icon,40)}<span><strong>${ai.name} <em>${ai.label}</em></strong><small>${ai.description}</small></span></label>`).join('');
 openDialog('选择这一季的对手',`<p class="dialog-lead">双方各有 $3,000 和一片农田。30 天后，现金更多的一方获胜。</p><div class="ai-choices">${cards}</div><label class="form-label">地图种子<input id="seed" type="number" min="0" max="2147483647" value="42"></label>${btn('start','开局对弈 '+icon('arrow',18),'primary-button')}`);
}
async function startNew() {
 if(busy||resting)return;
 const seed=Number($('#seed',dialog)?.value||42), opponent=$('input[name=opponent]:checked',dialog)?.value||'starter';
 if(!Number.isInteger(seed)||seed<0||seed>2147483647)return toast('请输入有效的整数种子');
 busy=true; const button=$('[data-do="start"]',dialog); if(button){button.disabled=true;button.textContent='正在准备农场…';}
 try { game=await api('new',{seed,opponent});remember();actions={};orders=[];unit=0;farmView=0;selected=[4,4];dialog.close();toast('新赛季开始。愿你收获满满。'); }
 catch(e){toast(e.message);if(button){button.disabled=false;button.textContent='重新尝试';}}
 finally{busy=false;render();}
}
function showPlant() {
 const t=own().tiles[pos()[1]][pos()[0]];
 if(t!==null && !(continuous && queueDriver.running))return toast('请先移动到空闲且已解锁的土地上。');
 openDialog('选择要播种的作物',`<p class="dialog-lead">${continuous&&queueDriver.running?'播种将在之前的行动完成后，于农夫脚下执行。':`种植位置 (${pos().join(', ')})。`}</p><div class="seed-options">${crops.map(k=>btn('plant-'+k,produce(k,42)+`<span><strong>${names[k]}</strong><small>种子 ${obs().private.seeds[k]||0} 颗 · ${ages[k]} 天起收获</small></span>`,'seed-option',!obs().private.seeds[k]&&!(continuous&&queueDriver.running)?'disabled':'')).join('')}</div><p class="subtle">种子要先成交才能播种。可随移动一起买，也可以点「只交易」。</p>`);
}
function showMore() {
 const opts=[['FERTILIZE','施肥'],['BUILD_COOP','建造鹅舍'],['BUILD_PASTURE','建造牧场'],['FEED','喂养动物'],['CARE','照料动物'],['COLLECT_FERTILIZER','收集肥料'],['PASS','休息一回合']];
 openDialog('农场里的更多事情',`<div class="more-actions">${opts.map(([k,v])=>btn('op-'+k,v,'outline-button')).join('')}</div><h3>仓库与动物</h3><label class="form-label">物品<select id="item">${Object.keys(names).map(k=>`<option value="${k}">${names[k]} · 仓库 ${obs().private.shed[k]||0} / 背包 ${bag()[k]||0}</option>`).join('')}</select></label><div class="button-row">${btn('pickup','从仓库取出 1 份','outline-button')}${btn('place','放置 1 份','outline-button')}</div><p class="subtle">取出货物需站在中央仓库旁；放置动物需站在对应的空建筑上。喂养与施肥消耗背包里的物品。</p><h3>多一些土地，多一双手</h3><div class="button-row">${btn('hire','雇佣一名农工','outline-button')}${btn('land','购买下一块土地','outline-button')}</div><p class="subtle">加入市场订单，推进回合后生效。新农工从下一回合开始行动，日末离开。</p>`);
}
function showResult() {
 const a=own().money,b=obs().farms[1].money;
 openDialog(a>b?'这一季，你收获了胜利。':a===b?'一样丰盛的两个赛季。':'这一季的经验，留给下一次丰收。',`<div class="result-art">${produce('WHEAT',100)}</div><div class="result-scores"><div><small>你的最终现金</small><strong>$${money(a)}</strong></div><div><small>对手最终现金</small><strong>$${money(b)}</strong></div></div><p class="dialog-lead">${a>b?'恭喜！你比对手积累了更多现金。':a===b?'平局。两座农场积累了相同的现金。':'对手赢得了本局。试着平衡种植、浇水和出售的时间。'}</p>${btn('new','开启新赛季 '+icon('arrow',16),'primary-button')}`);
}
async function handle(action) {
 if(action==='close'){dialog.close();return;}
 if(action==='stop-queue'){queueDriver.cancel();return;}
 if(action==='mode'){
  if(busy&&!queueDriver.running)return toast('请等待当前回合结算');
  if(!continuous && (orders.length||Object.keys(actions).length)){toast('请先提交或清空已计划的回合');return;}
  if(continuous)queueDriver.cancel();
  continuous=!continuous;render();return;
 }
 if(resting || (busy && !(continuous && queueDriver.running)))return;
 if(queueDriver.running && ['new','start','confirm-wait','wait','step'].includes(action))return toast('先按 Esc 停止队列，再执行此操作。');
 if(action==='rules')return showRules();
 if(action==='model')return showModel();
 if(action==='new')return showNew();
 if(action==='start')return startNew();
 if(action==='result')return showResult();
 if(action==='home'){section='farm';render();return;}
 if(action==='market'||action==='inventory'){section=action;render();return;}
 if(action==='view-own'||action==='view-other'){farmView=action==='view-own'?0:1;render();return;}
 if(action.startsWith('tab-')){tab=action.slice(4);render();return;}
 if(action==='toggle-zoom'){zoomed=!zoomed;render();return;}
 if(action==='toggle-grid'){gridVisible=!gridVisible;render();return;}
 if(action==='scene-help')return openDialog('这片农场，由你来经营。', '<p class="dialog-lead">点击地块选择目的地，每回合前进一步。连续模式下，农夫会自动走到目的地，农具箱选择的动作随后执行。Esc 清空后续动作。计划模式可组合多位农工的行动与订单，再一起提交。</p><p class="dialog-lead">可操作区域为右侧 10×10 农田。小屋、池塘和树林为场景装饰。中央木箱是仓库；米色路牌标记尚未解锁的田地。</p><p class="subtle">方向键 / WASD 移动 · 1–6 选择工具 · 空格推进回合</p>');
 if(action==='credits')return openDialog('像素里的小小世界', '<p class="dialog-lead">场景使用 Kenney 的 Tiny Farm 与 Tiny Town 像素素材（CC0），搭配为本项目绘制的池塘、田地和界面。</p><div class="credit-links"><a href="https://kenney.nl/assets/tiny-farm" target="_blank" rel="noreferrer">Kenney · Tiny Farm ↗</a><a href="https://kenney.nl/assets/tiny-town" target="_blank" rel="noreferrer">Kenney · Tiny Town ↗</a><a href="https://github.com/elliotfontaine/untitled-farming-sim" target="_blank" rel="noreferrer">设计参考 · Seasons Gone ↗</a></div><p class="subtle">部分商品图标为近似表示，具体品类与规则以文字为准。</p>');
 if(action==='clear-orders'){orders=[];render();return;}
 if(action==='toggle-log'){logOpen=!logOpen;render();return;}
 if(action==='export'){
  try {const data=await api('export?id='+game.id), url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download=`fieldnote-${game.seed}-turn-${obs().step}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);toast('对局记录已导出');}catch(e){toast(e.message);}return;
 }
 if(game.done)return toast('这一季已经结束，开启新赛季继续吧。');
 if(action==='step')return continuous?plan(['PASS']):advance();
 if(action==='clear-action'){delete actions[unit];render();return;}
 if(action.startsWith('remove-')){orders.splice(Number(action.slice(7)),1);render();return;}
 if(action.startsWith('trade-')){
  const n=Number($('#quantity').value);if(!Number.isInteger(n)||n<1||n>100)return toast('每笔请输入 1–100 的整数数量');
  const k=action.slice(6);return queue([tab==='seed'?'BUY_SEED':tab==='sell'?'SELL':['GOOSE','COW','SHEEP'].includes(k)?'BUY_ANIMAL':'BUY_PRODUCT',k,n]);
 }
 if(action.startsWith('move-')) {
  const dir=action.slice(5),[x,y]=pos();
  if((dir==='NORTH'&&y===0)||(dir==='SOUTH'&&y===9)||(dir==='WEST'&&x===0)||(dir==='EAST'&&x===9))return toast('已经到农场边缘啦。');
  const delta={NORTH:[0,-1],SOUTH:[0,1],WEST:[-1,0],EAST:[1,0]}[dir];selected=[x+delta[0],y+delta[1]];return plan([dir]);
 }
 if(action==='plant')return showPlant();
 if(action.startsWith('plant-')){crop=action.slice(6);const count=Object.entries(actions).filter(([k,v])=>Number(k)!==unit&&v[0]==='PLANT'&&v[1]===crop).length;if(!(continuous&&queueDriver.running)&&count>=obs().private.seeds[crop])return toast('其他农工已计划用完这些种子，请调整行动。');dialog.close();return plan(['PLANT',crop]);}
 if(action==='water'){if(continuous&&queueDriver.running)return plan(['WATER']);const t=own().tiles[pos()[1]][pos()[0]];if(t?.kind!=='PLANT')return toast('农夫脚下没有作物，先移动到需要浇水的地块。');if(t.watered_today)return toast('这块作物今天已经喝够水啦。');return plan(['WATER']);}
 if(action==='harvest'){if(continuous&&queueDriver.running)return plan(['HARVEST']);const t=own().tiles[pos()[1]][pos()[0]];if(!t?.yield_units||t.kind==='PLANT'&&obs().day-t.planted_day<ages[t.crop])return toast('农夫脚下还没有可以收获的作物或动物产品。');return plan(['HARVEST']);}
 if(action==='dig')return plan(['DIG']);
 if(action==='drop'){if(!shedAccess())return toast('请先走到中央仓库旁的四个入口格之一。');return plan(['DROP']);}
 if(action==='more')return showMore();
 if(action.startsWith('op-')){dialog.close();return plan([action.slice(3)]);}
 if(action==='pickup'||action==='place'){const item=$('#item',dialog).value;if(action==='pickup'&&!shedAccess())return toast('从仓库取出货物，需要先走到中央仓库旁。');dialog.close();return plan([action==='pickup'?'PICKUP':'PLACE',item,1]);}
 if(action==='hire'||action==='land'){dialog.close();return queue([action==='hire'?'HIRE':'BUY_LAND']);}
 if(action==='wait')return openDialog('让农场，休息到明天。',`<p class="dialog-lead">将跳过今天剩余 ${Math.min(24-obs().hour,game.totalTurns-obs().step)} 个回合。你的农夫和农工休息，对手继续行动。</p><div class="rule-callout"><div><strong>请先确认今天的作物都已浇水。</strong><p>跳过不会自动种植、浇水或出售；未提交的行动和订单会清空。日末背包自动入仓，农工离开。可以随时停止连续休息。</p></div></div>${btn('confirm-wait','开始休息','primary-button')}`);
 if(action==='confirm-wait'){
  const day=obs().day;actions={};orders=[];dialog.close();resting=true;
  let stopped=false;openDialog('风吹麦浪，时间慢慢走。',`<p id="wait-progress">正在休息…</p><button id="stop-wait" class="outline-button">停止休息</button>`);
  $('#stop-wait').onclick=()=>{stopped=true;};
  while(!stopped&&dialog.open&&!game.done&&obs().day===day){const step=obs().step;await advance({farmer:['PASS'],hands:[],market:[]});if(obs().step===step)break;if($('#wait-progress'))$('#wait-progress').textContent=`第 ${obs().day+1} 天 · ${String(obs().hour).padStart(2,'0')}:00`;await new Promise(r=>setTimeout(r,70));}
  resting=false;if(!game.done)dialog.close();return;
 }
}
document.addEventListener('click', e=>{
 const b=e.target.closest('[data-do]');if(b){e.preventDefault();handle(b.dataset.do);return;}
 const tile=e.target.closest('.map-tile');
 if(tile&&(!busy||continuous&&queueDriver.running)&&!resting){
  selected=[Number(tile.dataset.x),Number(tile.dataset.y)];
  if(farmView===0&&!game.done&&continuous){queueDriver.push({unit,to:[...selected]});return;}
  if(farmView===0&&!game.done){const [x,y]=pos(),[tx,ty]=selected;if(x!==tx||y!==ty){actions[unit]=[tx!==x?(tx>x?'EAST':'WEST'):(ty>y?'SOUTH':'NORTH')];}}
  render();
 }
});
document.addEventListener('change',e=>{
 if(e.target.id==='unit'){unit=Number(e.target.value);selected=[...pos()];render();}
 if(e.target.id==='quantity')marketQuantity=Number(e.target.value);
});
document.addEventListener('keydown',e=>{
 if(e.key==='Escape' && queueDriver.running){queueDriver.cancel();return;}
 if(e.target.closest('input,select,textarea')||dialog.open||!game||busy&&!queueDriver.running||resting||e.repeat)return;
 if(e.target.closest('.map-tile')&&(e.key==='Enter'||e.key===' ')){e.preventDefault();e.target.closest('.map-tile').dispatchEvent(new MouseEvent('click',{bubbles:true}));return;}
 const dirs={ArrowUp:'NORTH',ArrowDown:'SOUTH',ArrowLeft:'WEST',ArrowRight:'EAST',w:'NORTH',s:'SOUTH',a:'WEST',d:'EAST'};
 if(dirs[e.key]){e.preventDefault();handle('move-'+dirs[e.key]);return;}
 const shortcuts={'1':'plant','2':'water','3':'harvest','4':'dig','5':'drop','6':'more'};
 if(shortcuts[e.key]){e.preventDefault();handle(shortcuts[e.key]);return;}
 if(e.key===' '&&!e.target.closest('button,a')){e.preventDefault();handle('step');}
});
dialog.addEventListener('click',e=>{if(e.target===dialog){const r=dialog.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)dialog.close();}if(e.target.closest('[data-do^="move-"]'))dialog.close();});
async function boot(){
 try{health=await api('health');let id;try{id=localStorage.getItem('farm-rl-session');}catch{}if(id){try{game=await api('game?id='+encodeURIComponent(id));}catch{}}
 const firstVisit=!game; if(firstVisit)game=await api('new',{seed:42,opponent:'starter'});remember();render();if(firstVisit)showNew();
 }catch(e){app.innerHTML=`<div class="loading-screen"><span class="loading-leaf">❋</span><h1>农场还没醒来</h1><p>${esc(e.message)}</p><p>双击游戏目录里的「开始游戏.cmd」，再打开游戏。</p><button class="outline-button" id="retry">重新连接</button></div>`;$('#retry').onclick=boot;}
}
boot();
