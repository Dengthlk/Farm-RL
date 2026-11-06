// Kenney Tiny Farm + Tiny Town, CC0. See assets/CREDITS.md.
const asset = (pack, id) => `/assets/${pack}/Tiles/tile_${String(id).padStart(4,'0')}.png`;
export const sprite = (id, size=32, pack='tiny-farm', cls='') => `<img class="pixel-sprite ${cls}" src="${asset(pack,id)}" width="${size}" height="${size}" alt="" draggable="false">`;
const items = {CARROT:8,WHEAT:68,TOMATO:44,STRAWBERRY:44,MELON:56,EGG:125,MILK:123,WOOL:120,FERTILIZER:74,GOOSE:122,COW:121,SHEEP:120};
export function itemSprite(type,size=40) {return sprite(items[type]??74,size,'tiny-farm',type==='STRAWBERRY'?'berry-sprite':'');}
const image = (id,x,y,w=16,h=w,pack='tiny-town',cls='') => `<image class="pixel-art ${cls}" href="${asset(pack,id)}" x="${x}" y="${y}" width="${w}" height="${h}" image-rendering="pixelated"/>`;
const tileXY=(x,y)=>[192+x*16+(x>=5?16:0),48+y*16+(y>=5?16:0)];
const tree=(x,y,type=0)=>{
 if(type===1)return image(4,x,y,24,32);
 let out='';
 for(let row=0;row<3;row++)for(let col=0;col<3;col++)out+=image([6,7,8,18,19,20,30,31,32][row*3+col],x+col*16,y+row*16);
 return out;
};
function building(x,y,red=false){
 let out='';const top=red?[52,53,53,54]:[48,49,49,50],bottom=red?[64,65,65,66]:[60,61,61,62];
 [top,bottom,[72,73,73,75],[84,73,85,75]].forEach((row,ry)=>row.forEach((n,rx)=>out+=image(n,x+rx*16,y+ry*16)));
 return out;
}
function scenery(){
 let s='<rect width="432" height="288" fill="#82be65"/>';
 // Deterministic ground detail, so the world does not jump between turns.
 for(let y=0;y<18;y++)for(let x=0;x<27;x++){
  const n=(x*7+y*13)%17;
  if(n<5)s+=image(n===0?2:1,x*16,y*16);
 }
 // Garden paths connect the house, field and pond.
 s+='<path d="M100 111V137H280V258H96M160 137V258M280 137V36" fill="none" stroke="#bc9055" stroke-width="20"/>';
 s+='<path d="M100 111V137H280V258H96M160 137V258M280 137V36" fill="none" stroke="#e3b878" stroke-width="16"/>';
 for(let i=0;i<30;i++){let x=92+(i*13)%200,y=i%2?134:255;s+=`<rect x="${x}" y="${y}" width="2" height="1" fill="#cba46c"/>`;}
 // A stepped, pixel-edged pond with animated light on the water.
 s+='<path d="M32 173h16v-8h40v8h16v12h8v40h-8v12H88v8H48v-8H32v-16h-8v-32h8Z" fill="#598a54"/><path d="M36 177h16v-8h32v8h16v12h8v32h-8v12H84v8H52v-8H36v-16h-8v-24h8Z" fill="#b4d6a0"/><path d="M40 181h16v-8h24v8h16v12h8v24h-8v12H80v8H56v-8H40v-16h-8v-16h8Z" fill="#74b6bc"/><path d="M40 183h17v-7h23v7h15v12h6v18h-6v12H81v8H57v-8H43v-13h-8v-15h5Z" fill="#8ccbd0"/><g class="water-shimmer" fill="#d0ebd6"><path d="M49 191h14v2H49zm26 20h17v2H75zm-25 10h10v2H50zm29-35h9v2h-9Z"/></g><path d="M47 205h8v-3h5v7H47Z" fill="#4d9876"/><rect x="53" y="201" width="3" height="3" fill="#f5d69e"/>';
 for(let i=0;i<3;i++)s+=image(98,106+i*16,200,16,16,'tiny-farm');
 s+=building(66,48,true);
 s+=image(55,91,55); // skylight
 s+=image(83,133,113); // signpost
 s+=image(106,136,75,24,24); // well
 s+=image(43,92,112)+image(43,96,122);
 s+=image(94,50,110)+image(115,48,131)+image(57,127,148);
 // Little supply yard, apart from the playable grid.
 s+=image(96,113,228,16,16,'tiny-farm')+image(97,129,228,16,16,'tiny-farm');
 s+=image(90,113,244,16,16,'tiny-farm')+image(91,129,244,16,16,'tiny-farm');
 s+=image(73,146,240,16,16,'tiny-farm');
 // Perimeter fencing and a gate, kept outside playable tiles.
 for(let x=176;x<=384;x+=16){if(x<272||x>288)s+=image(45,x,27);if(x<272||x>288)s+=image(45,x,232);}
 s+=image(44,168,27)+image(46,392,27)+image(44,168,232)+image(46,392,232);
 for(let y=43;y<232;y+=16){s+=image(56,173,y);s+=image(58,392,y);}
 // Forest frame, dense at the back and sparse by the front gate.
 [[0,-18],[34,-26],[68,-20],[104,-30],[140,-26],[182,-33],[224,-29],[315,-31],[354,-22],[391,-15],[-20,25],[-22,66],[-19,112],[398,34],[408,80],[403,127],[399,173],[-23,244],[8,259],[44,264],[330,265],[371,245],[404,232]].forEach(([x,y],i)=>s+=tree(x,y));
 [[39,46],[48,148],[135,176],[148,202],[114,15],[379,116],[381,207],[315,250]].forEach(([x,y],i)=>s+=tree(x,y,1));
 [[42,128],[119,180],[116,196],[48,250],[333,19],[373,251],[24,156]].forEach(([x,y],i)=>s+=image(i%3===0?28:5,x,y));
 s+=image(78,146,176,16,16,'tiny-farm')+image(78,146,192,16,16,'tiny-farm');
 s+='<g class="butterfly" fill="#f6e0a0"><path d="M143 162h3v3h-3zm4 0h3v3h-3Z"/></g>';
 return s;
}
const background=scenery();
export function farmScene({farm,selected,unit=0,view=0,day=0,hour=0,actions={},motion=null,showGrid=false,previousPositions=[]}) {
 let plots='',plants='',units='',locked='',route='';
 for(let y=0;y<10;y++)for(let x=0;x<10;x++){
  const t=farm.tiles[y][x], [px,py]=tileXY(x,y), isLocked=t==='LOCKED';
  const name=isLocked?'未解锁':t?.crop?({CARROT:'胡萝卜',WHEAT:'小麦',TOMATO:'番茄',STRAWBERRY:'草莓',MELON:'甜瓜'}[t.crop]):t?.kind==='WEED'?'杂草':t?.animal?'牲畜':'空地';
  if(!isLocked){
   plots+=`<rect x="${px}" y="${py}" width="16" height="16" fill="${t?.watered_today?'#9b6b4a':'#bc8755'}"/><path d="M${px} ${py+3}h16m-16 7h16" stroke="${t?.watered_today?'#82563f':'#a87348'}" stroke-width="2"/><path d="M${px+1} ${py+5}h14m-14 7h14" stroke="${t?.watered_today?'#ad7b53':'#ce995f'}" stroke-width="1"/>`;
  }else{
   plots+=`<rect x="${px}" y="${py}" width="16" height="16" fill="${(x+y)%2?'#91bd6d':'#95c171'}"/>`;
   if((x*3+y*7)%4===0)plots+=image(1,px,py);
  }
  if(t?.kind==='PLANT'){
   const age=day-t.planted_day, first={CARROT:2,WHEAT:2,TOMATO:8,STRAWBERRY:10,MELON:10}[t.crop];
   const stage=Math.min(2,Math.floor(age/Math.max(1,first/2)));
   const grow={CARROT:[4,5,6],WHEAT:[64,65,66],TOMATO:[40,41,42],STRAWBERRY:[40,41,42],MELON:[52,53,54]};
   plants+=image(grow[t.crop][stage],px,py-2,16,16,'tiny-farm',t.crop==='STRAWBERRY'?'berry-sprite':'');
   if(!t.watered_today)plants+=`<path d="m${px+13} ${py+1}-2 3v2h4V${py+4}Z" fill="#a7e3e8" stroke="#397e91" stroke-width=".5"/>`;
   if(age>=first&&t.yield_units>0)plants+=`<rect class="ready-spark" x="${px+1}" y="${py}" width="2" height="2" fill="#ffe5a3"/>`;
  }
  if(t?.kind==='WEED')plants+=image(80,px,py,16,16,'tiny-farm');
  if(t?.kind==='COOP'||t?.kind==='PASTURE')plants+=image(t.animal?({GOOSE:122,COW:121,SHEEP:120}[t.animal]):90,px,py-3,16,16,'tiny-farm');
  plots+=`<rect class="map-tile" data-x="${x}" data-y="${y}" x="${px}" y="${py}" width="16" height="16" fill="transparent" stroke="${showGrid?'#50643e88':isLocked?'transparent':'#85613c44'}" stroke-width=".5" tabindex="0" role="button" aria-label="地块 ${x}, ${y}，${name}"/>`;
 }
 [['NE',288,48,1000],['SW',192,144,2000],['SE',288,144,4000]].forEach(([q,x,y,cost])=>{
  if(!farm.unlocked_quadrants.includes(q)){
   // Small edge vegetation makes locked plots read as meadow, not grayed-out UI.
   locked+=image(5,x+57,y+9)+image(1,x+10,y+51)+image(5,x+8,y+10);
   locked+=`<g class="land-sign" pointer-events="none"><rect x="${x+21}" y="${y+34}" width="44" height="21" fill="#486345" opacity=".9"/><rect x="${x+23}" y="${y+32}" width="40" height="21" fill="#f5e7bd"/><path d="M${x+39} ${y+36}v-2a3 3 0 0 1 6 0v2m-7 0h8v6h-8Z" fill="#8f845d"/><text x="${x+43}" y="${y+49}" text-anchor="middle" fill="#75694b" font-size="5" font-family="monospace">${q} · ${cost.toLocaleString()}</text></g>`;
  }
 });
 const all=[farm.farmer,...farm.hands];
 all.forEach(([x,y],i)=>{
  const [px,py]=tileXY(x,y), old=previousPositions[i];
  const playing=motion?.unit===i && view===0 ? motion : null;
  const origin=playing?.start || old;
  const [ox,oy]=origin?tileXY(origin[0],origin[1]):[px,py];
  const row=playing?playing.clip*4+playing.face:0;
  const delay=playing?-Math.max(0,performance.now()-playing.time):0;
  const clock=playing?`animation:farmer-frames ${playing.duration}ms steps(11,end) ${delay}ms both`:'';
  const character=`<svg x="-8" y="-10" width="32" height="32" viewBox="0 0 40 40" overflow="hidden"><image class="pixel-art" href="/assets/Farmer-actions.png" x="0" y="${-row*40}" width="480" height="800" style="${clock}"/></svg>`;
  units+=`<g class="unit-position" style="--from-x:${ox-px}px;--from-y:${oy-py}px;${playing?`animation-duration:${playing.duration}ms;animation-delay:${delay}ms`:''}" transform="translate(${px} ${py})"><ellipse cx="8" cy="13" rx="6" ry="2" fill="#3f4829" opacity=".3"/>${i===unit&&view===0?'<path class="farmer-pointer" d="M5-9h6L8-5Z" fill="#fff1b7" stroke="#6c7348" stroke-width=".5"/>':''}${character}${i?`<text x="13" y="14" font-size="5" fill="#fff3ca" stroke="#4d563b" stroke-width=".3">${i}</text>`:''}</g>`;
 });
 const [sx,sy]=tileXY(...selected);
 const active=all[unit]||farm.farmer;
 if(view===0&&(selected[0]!==active[0]||selected[1]!==active[1])){
  const [ax,ay]=tileXY(...active);
  route=`<path d="M${ax+8} ${ay+8}H${sx+8}V${sy+8}" fill="none" stroke="#fff4bf" stroke-width="1" stroke-dasharray="2 2" opacity=".85"/>`;
 }
 const cursor=`<path class="selection-corners" d="M${sx-1} ${sy+4}v-5h5m7 0h6v5m0 7v6h-6m-7 0h-5v-6" fill="none" stroke="#fff2b6" stroke-width="1.5"/>`;
 const shed=image(126,271,127,18,18,'tiny-farm');
 const night=hour>=19||hour<5;
 return `<svg class="farm-map" viewBox="0 0 432 288" role="group" aria-label="可交互像素农场地图" shape-rendering="crispEdges"><g class="world-ground">${background}</g>${plots}<g pointer-events="none">${locked}${plants}${shed}${route}${cursor}${units}</g><g pointer-events="none" class="ambient"><path class="cloud-shadow" d="M-30 40h30v-8h50v8h30v20H50v8H0v-8h-30Z" fill="#fff8dd" opacity=".09"/>${night?'<rect width="432" height="288" fill="#3d4b91" opacity=".08"/>':''}</g></svg>`;
}
