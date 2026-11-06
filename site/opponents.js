export const opponents = [
  {id:'starter', name:'Starter', label:'官方入门', icon:109, description:'按固定策略种胡萝卜，适合熟悉第一季的操作。'},
  {id:'bc', name:'BC', label:'行为克隆', icon:108, description:'使用回放训练出的策略，在本机决定每回合的农活与交易。'},
  {id:'ppo', name:'PPO', label:'自博弈 · 1200 轮', icon:109, description:'使用第 1200 轮 PPO 权重，与你竞争赛季末的现金。'}
];
export const opponentName = id => opponents.find(opponent=>opponent.id===id).name;

